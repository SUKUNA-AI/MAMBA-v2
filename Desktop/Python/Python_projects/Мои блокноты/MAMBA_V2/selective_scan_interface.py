import torch
import torch.nn.functional as F
from custom_amp_decorator import custom_bwd, custom_fwd

try:
    from causal_conv1d import causal_conv1d_fn
    import causal_conv1d_cuda
except ImportError:
    causal_conv1d_fn = None
    causal_conv1d_cuda = None

from mamba_ssm.ops.triton.layer_norm import _layer_norm_fwd
import selective_scan_cuda


class SelectiveScanFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                return_last_state=False):
        """Forward pass для Selective Scan."""
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if z is not None and z.stride(-1) != 1:
            z = z.contiguous()

        # Замена rearrange(B, "b dstate l -> b 1 dstate l")
        if B.dim() == 3:
            B = B.unsqueeze(1)
            ctx.squeeze_B = True

        # Замена rearrange(C, "b dstate l -> b 1 dstate l")
        if C.dim() == 3:
            C = C.unsqueeze(1)
            ctx.squeeze_C = True

        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus)
        ctx.delta_softplus = delta_softplus
        ctx.has_z = z is not None
        last_state = x[:, :, -1, 1::2]  # (batch, dim, dstate)

        if not ctx.has_z:
            ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
            return out if not return_last_state else (out, last_state)
        else:
            ctx.save_for_backward(u, delta, A, B, C, D, z, delta_bias, x, out)
            out_z = rest[0]
            return out_z if not return_last_state else (out_z, last_state)

    @staticmethod
    def backward(ctx, dout, *args):
        """Backward pass для Selective Scan."""
        if not ctx.has_z:
            u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
            z = None
            out = None
        else:
            u, delta, A, B, C, D, z, delta_bias, x, out = ctx.saved_tensors

        if dout.stride(-1) != 1:
            dout = dout.contiguous()

        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, z, delta_bias, dout, x, out, None, ctx.delta_softplus, False
        )
        dz = rest[0] if ctx.has_z else None

        # Замена squeeze для B и C
        if getattr(ctx, "squeeze_B", False):
            dB = dB.squeeze(1)
        if getattr(ctx, "squeeze_C", False):
            dC = dC.squeeze(1)

        return (du, ddelta, dA, dB, dC,
                dD if D is not None else None,
                dz,
                ddelta_bias if delta_bias is not None else None,
                None,
                None)


def rms_norm_forward(x, weight, bias, eps=1e-6, is_rms_norm=True):
    """RMS нормализация."""
    if x.stride(-1) != 1:
        x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    y = _layer_norm_fwd(x, weight, bias, eps, None, residual_dtype=None, is_rms_norm=is_rms_norm)[0]
    return y


def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                      return_last_state=False):
    """Обёртка для SelectiveScanFn."""
    return SelectiveScanFn.apply(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state)


def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                       return_last_state=False):
    """Эталонная реализация Selective Scan."""
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)

    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3

    if A.is_complex():
        if is_variable_B:
            # Замена rearrange(B.float(), "... (L two) -> ... L two", two=2)
            B = torch.view_as_complex(B.float().reshape(*B.shape[:-1], -1, 2))
        if is_variable_C:
            C = torch.view_as_complex(C.float().reshape(*C.shape[:-1], -1, 2))
    else:
        B = B.float()
        C = C.float()

    x = A.new_zeros((batch, dim, dstate))
    ys = []
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))

    if not is_variable_B:
        deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
    else:
        if B.dim() == 3:
            deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
        else:
            # Замена repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
            H = dim // B.shape[1]
            B = B.unsqueeze(2).expand(-1, -1, H, -1, -1).reshape(B.shape[0], B.shape[1] * H, B.shape[2], B.shape[3])
            deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)

    if is_variable_C and C.dim() == 4:
        H = dim // C.shape[1]
        C = C.unsqueeze(2).expand(-1, -1, H, -1, -1).reshape(C.shape[0], C.shape[1] * H, C.shape[2], C.shape[3])

    last_state = None
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if not is_variable_C:
            y = torch.einsum('bdn,dn->bd', x, C)
        else:
            if C.dim() == 3:
                y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
            else:
                y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
        if i == u.shape[2] - 1:
            last_state = x
        if y.is_complex():
            y = y.real * 2
        ys.append(y)

    y = torch.stack(ys, dim=2)  # (batch dim L)
    # Замена rearrange(D, "d -> d 1")
    out = y if D is None else y + u * D.unsqueeze(-1)
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state)


class MambaInnerFn(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
                out_proj_weight, out_proj_bias, A, B=None, C=None, D=None, delta_bias=None,
                B_proj_bias=None, C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1,
                b_rms_weight=None, c_rms_weight=None, dt_rms_weight=None, b_c_dt_rms_eps=1e-6):
        """Forward pass для Mamba."""
        assert causal_conv1d_cuda is not None, "causal_conv1d_cuda is not available."
        assert checkpoint_lvl in [0, 1]

        L = xz.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)

        if torch.is_autocast_enabled():
            x_proj_weight = x_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            delta_proj_weight = delta_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            out_proj_weight = out_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            out_proj_bias = (out_proj_bias.to(dtype=torch.get_autocast_gpu_dtype())
                             if out_proj_bias is not None else None)

        if xz.stride(-1) != 1:
            xz = xz.contiguous()

        # Замена rearrange(conv1d_weight, "d 1 w -> d w")
        conv1d_weight = conv1d_weight.reshape(conv1d_weight.shape[0], -1)
        x, z = xz.chunk(2, dim=1)
        conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None
        conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(
            x, conv1d_weight, conv1d_bias, None, None, None, True
        )

        # Замена rearrange(conv1d_out, 'b d l -> (b l) d')
        x_dbl = F.linear(conv1d_out.reshape(-1, conv1d_out.shape[-1]), x_proj_weight)  # (bl d)
        delta = delta_proj_weight @ x_dbl[:, :delta_rank].t()
        # Замена rearrange(... "d (b l) -> b d l", l=L)
        delta = delta.reshape(delta.shape[0], -1, L).transpose(1, 2)  # (b l d) -> (b d l)

        ctx.is_variable_B = B is None
        ctx.is_variable_C = C is None
        ctx.B_proj_bias_is_None = B_proj_bias is None
        ctx.C_proj_bias_is_None = C_proj_bias is None

        if B is None:  # variable B
            B = x_dbl[:, delta_rank:delta_rank + d_state]  # (bl dstate)
            if B_proj_bias is not None:
                B = B + B_proj_bias.to(dtype=B.dtype)
            if not A.is_complex():
                # Замена rearrange(B, "(b l) dstate -> b 1 dstate l", l=L)
                B = B.reshape(-1, L, d_state).transpose(1, 2).unsqueeze(1)  # (b 1 dstate l)
            else:
                # Замена rearrange(B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2)
                B = B.reshape(-1, L, d_state, 2).transpose(1, 2).unsqueeze(1)  # (b 1 dstate (l 2))

        else:
            if B.stride(-1) != 1:
                B = B.contiguous()

        if C is None:  # variable C
            C = x_dbl[:, -d_state:]  # (bl dstate)
            if C_proj_bias is not None:
                C = C + C_proj_bias.to(dtype=C.dtype)
            if not A.is_complex():
                # Замена rearrange(C, "(b l) dstate -> b 1 dstate l", l=L)
                C = C.reshape(-1, L, d_state).transpose(1, 2).unsqueeze(1)  # (b 1 dstate l)
            else:
                # Замена rearrange(C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2)
                C = C.reshape(-1, L, d_state, 2).transpose(1, 2).unsqueeze(1)  # (b 1 dstate (l 2))

        else:
            if C.stride(-1) != 1:
                C = C.contiguous()

        if D is not None:
            D = D.contiguous()

        # RMS нормализация для B, C, delta
        if b_rms_weight is not None:
            # Замена rearrange(B, "b 1 dstate l -> (b l) dstate")
            B = B.squeeze(1).reshape(-1, d_state)
            B = rms_norm_forward(B, b_rms_weight, bias=None, eps=b_c_dt_rms_eps)
            # Замена rearrange(B, "(b l) dstate -> b 1 dstate l", l=L)
            B = B.reshape(B.shape[0] // L, L, d_state).unsqueeze(1)

        if c_rms_weight is not None:
            C = C.squeeze(1).reshape(-1, d_state)
            C = rms_norm_forward(C, c_rms_weight, bias=None, eps=b_c_dt_rms_eps)
            C = C.reshape(C.shape[0] // L, L, d_state).unsqueeze(1)

        if dt_rms_weight is not None:
            # Замена rearrange(delta, "b d l -> (b l) d")
            delta = delta.reshape(-1, delta.shape[-1])
            delta = rms_norm_forward(delta, dt_rms_weight, bias=None, eps=b_c_dt_rms_eps)
            # Замена rearrange(delta, "(b l) d -> b d l", l=L)
            delta = delta.reshape(delta.shape[0] // L, L, -1).transpose(1, 2)

        out, scan_intermediates, out_z = selective_scan_cuda.fwd(
            conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus
        )

        ctx.delta_softplus = delta_softplus
        ctx.out_proj_bias_is_None = out_proj_bias is None
        ctx.checkpoint_lvl = checkpoint_lvl
        ctx.b_rms_weight = b_rms_weight
        ctx.c_rms_weight = c_rms_weight
        ctx.dt_rms_weight = dt_rms_weight
        ctx.b_c_dt_rms_eps = b_c_dt_rms_eps

        if checkpoint_lvl >= 1:
            conv1d_out, delta = None, None

        ctx.save_for_backward(xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight,
                              delta_proj_weight, out_proj_weight, conv1d_out, delta,
                              A, B, C, D, delta_bias, scan_intermediates, b_rms_weight,
                              c_rms_weight, dt_rms_weight, out)

        # Замена rearrange(out_z, "b d l -> b l d")
        return F.linear(out_z.transpose(1, 2), out_proj_weight, out_proj_bias)

    @staticmethod
    @custom_bwd
    def backward(ctx, dout):
        """Backward pass для Mamba."""
        assert causal_conv1d_cuda is not None, "causal_conv1d_cuda is not available."
        (xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight, delta_proj_weight, out_proj_weight,
         conv1d_out, delta, A, B, C, D, delta_bias, scan_intermediates, b_rms_weight,
         c_rms_weight, dt_rms_weight, out) = ctx.saved_tensors

        L = xz.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
        x, z = xz.chunk(2, dim=1)

        if dout.stride(-1) != 1:
            dout = dout.contiguous()

        if ctx.checkpoint_lvl == 1:
            conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(
                x, conv1d_weight, conv1d_bias, None, None, None, True
            )
            delta = delta_proj_weight @ x_dbl[:, :delta_rank].t()
            delta = delta.reshape(delta.shape[0], -1, L).transpose(1, 2)
            if dt_rms_weight is not None:
                delta = delta.reshape(-1, delta.shape[-1])
                delta = rms_norm_forward(delta, ctx.dt_rms_weight, None, ctx.b_c_dt_rms_eps)
                delta = delta.reshape(delta.shape[0] // L, L, -1).transpose(1, 2)
            if b_rms_weight is not None:
                B = B.squeeze(1).reshape(-1, d_state)
                B = rms_norm_forward(B, ctx.b_rms_weight, None, ctx.b_c_dt_rms_eps)
                B = B.reshape(B.shape[0] // L, L, d_state).unsqueeze(1)
            if c_rms_weight is not None:
                C = C.squeeze(1).reshape(-1, d_state)
                C = rms_norm_forward(C, ctx.c_rms_weight, None, ctx.b_c_dt_rms_eps)
                C = C.reshape(C.shape[0] // L, L, d_state).unsqueeze(1)

        dxz = torch.empty_like(xz)
        dx, dz = dxz.chunk(2, dim=1)

        # Замена rearrange(dout, "b l e -> e (b l)")
        dout = dout.transpose(1, 2).reshape(-1, dout.shape[1])
        # Замена rearrange(out_proj_weight.t() @ dout, "d (b l) -> b d l", l=L)
        dout_y = (out_proj_weight.t() @ dout).reshape(-1, L, out_proj_weight.shape[1]).transpose(1, 2)

        dconv1d_out, ddelta, dA, dB, dC, dD, ddelta_bias, dz, out_z = selective_scan_cuda.bwd(
            conv1d_out, delta, A, B, C, D, z, delta_bias, dout_y, scan_intermediates, out, dz,
            ctx.delta_softplus, True
        )

        # Замена rearrange(out_z, "b d l -> d (b l)")
        dout_proj_weight = torch.einsum("eB,dB->ed", dout, out_z.transpose(1, 2).reshape(out_z.shape[1], -1))
        dout_proj_bias = dout.sum(dim=(0, 1)) if not ctx.out_proj_bias_is_None else None
        dD = dD if D is not None else None

        dx_dbl = torch.empty_like(x_dbl)
        dB_proj_bias = None
        if ctx.is_variable_B:
            if not A.is_complex():
                # Замена rearrange(dB, "b 1 dstate l -> (b l) dstate")
                dB = dB.squeeze(1).reshape(-1, d_state)
            else:
                # Замена rearrange(dB, "b 1 dstate (l two) -> (b l) (dstate two)", two=2)
                dB = dB.squeeze(1).reshape(-1, d_state * 2)
            dB_proj_bias = dB.sum(0) if not ctx.B_proj_bias_is_None else None
            dx_dbl[:, delta_rank:delta_rank + d_state] = dB
            dB = None

        dC_proj_bias = None
        if ctx.is_variable_C:
            if not A.is_complex():
                dC = dC.squeeze(1).reshape(-1, d_state)
            else:
                dC = dC.squeeze(1).reshape(-1, d_state * 2)
            dC_proj_bias = dC.sum(0) if not ctx.C_proj_bias_is_None else None
            dx_dbl[:, -d_state:] = dC
            dC = None

        # Замена rearrange(ddelta, "b d l -> d (b l)")
        ddelta = ddelta.transpose(1, 2).reshape(ddelta.shape[1], -1)
        ddelta_proj_weight = torch.einsum("dB,Br->dr", ddelta, x_dbl[:, :delta_rank])
        dx_dbl[:, :delta_rank] = torch.einsum("dB,dr->Br", ddelta, delta_proj_weight)

        # Замена rearrange(dconv1d_out, "b d l -> d (b l)")
        dconv1d_out = dconv1d_out.transpose(1, 2).reshape(dconv1d_out.shape[1], -1)
        # Замена rearrange(conv1d_out, "b d l -> (b l) d")
        dx_proj_weight = torch.einsum("Br,Bd->rd", dx_dbl, conv1d_out.reshape(-1, conv1d_out.shape[-1]))
        dconv1d_out = torch.addmm(dconv1d_out, x_proj_weight.t(), dx_dbl.t(), out=dconv1d_out)
        # Замена rearrange(dconv1d_out, "d (b l) -> b d l", b=x.shape[0], l=x.shape[-1])
        dconv1d_out = dconv1d_out.reshape(x.shape[0], L, -1).transpose(1, 2)

        dx, dconv1d_weight, dconv1d_bias, *_ = causal_conv1d_cuda.causal_conv1d_bwd(
            x, conv1d_weight, conv1d_bias, dconv1d_out, None, None, None, dx, False, True
        )
        dconv1d_bias = dconv1d_bias if conv1d_bias is not None else None
        # Замена rearrange(dconv1d_weight, "d w -> d 1 w")
        dconv1d_weight = dconv1d_weight.reshape(dconv1d_weight.shape[0], 1, -1)

        return (dxz, dconv1d_weight, dconv1d_bias, dx_proj_weight, ddelta_proj_weight,
                dout_proj_weight, dout_proj_bias, dA, dB, dC, dD,
                ddelta_bias if delta_bias is not None else None,
                dB_proj_bias, dC_proj_bias, None, None, None, None, None, None)


def mamba_inner_fn(xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
                   out_proj_weight, out_proj_bias, A, B=None, C=None, D=None, delta_bias=None,
                   B_proj_bias=None, C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1,
                   b_rms_weight=None, c_rms_weight=None, dt_rms_weight=None, b_c_dt_rms_eps=1e-6):
    """Обёртка для MambaInnerFn."""
    return MambaInnerFn.apply(xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
                              out_proj_weight, out_proj_bias, A, B, C, D, delta_bias, B_proj_bias,
                              C_proj_bias, delta_softplus, checkpoint_lvl, b_rms_weight, c_rms_weight,
                              dt_rms_weight, b_c_dt_rms_eps)