import torch
import torch.nn as nn
import torch.nn.functional as F

class CalibrationModule(nn.Module):
    def __init__(
            self,
            d_model,
            lab_feature_dim,
            dropout=0.1,
            device='cuda',
            dtype=None,
    ):
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}

        # Laboratory data encoder
        self.lab_encoder = nn.Sequential(
            nn.Linear(lab_feature_dim, d_model, **factory_kwargs),
            nn.LayerNorm(d_model, eps=1e-5, **factory_kwargs),
            nn.GELU()
        )

        # Calibration gate
        self.calibration_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model, **factory_kwargs),
            nn.LayerNorm(d_model, eps=1e-5, **factory_kwargs),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(d_model, d_model, **factory_kwargs),
            nn.Sigmoid()
        )

        # Calibration transform
        self.calibration_transform = nn.Sequential(
            nn.Linear(d_model * 2, d_model, **factory_kwargs),
            nn.LayerNorm(d_model, eps=1e-5, **factory_kwargs),
            nn.Dropout(dropout),
            nn.GELU()
        )

    def forward(self, model_output, lab_data=None):
        """
        Apply calibration using laboratory data if available.

        Args:
            model_output: Model output tensor [batch, d_model]
            lab_data: Laboratory data tensor [batch, lab_feature_dim] or None

        Returns:
            Calibrated output tensor [batch, d_model]
        """
        if lab_data is None:
            return model_output

        # Encode laboratory data
        lab_encoded = self.lab_encoder(lab_data)

        # Compute calibration gate
        gate = self.calibration_gate(torch.cat([model_output, lab_encoded], dim=-1))

        # Apply gated fusion
        combined = torch.cat([model_output, lab_encoded], dim=-1)
        calibrated = model_output * gate + self.calibration_transform(combined) * (1 - gate)

        return calibrated
