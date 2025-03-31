import torch
import pandas as pd
import numpy as np
from model import PulpSieveModel

def load_model(checkpoint_path):
    """
    Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint

    Returns:
        Loaded model
    """
    model = PulpSieveModel.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model

def predict(model, telemetry_data, lab_data=None):
    """
    Make predictions using the model.

    Args:
        model: Trained model
        telemetry_data: Telemetry data tensor [batch, seq_len, input_dim]
        lab_data: Laboratory data tensor [batch, lab_feature_dim] or None

    Returns:
        If model.config.estimate_uncertainty is True:
            Tuple of (mean, variance) tensors [batch, output_dim]
        Else:
            Output tensor [batch, output_dim]
    """
    with torch.no_grad():
        return model(telemetry_data, lab_data)

def main():
    # Load model
    model = load_model('pulp_sieve_model.ckpt')

    # Load new telemetry data
    telemetry_data = pd.read_excel('new_telemetry.xlsx')

    # Convert to datetime and set as index
    telemetry_data['Время'] = pd.to_datetime(telemetry_data['Время'])
    telemetry_data.set_index('Время', inplace=True)

    # Prepare data for inference
    seq_len = model.config.seq_len

    # Get sequence of telemetry data
    telemetry_seq = []
    for i in range(len(telemetry_data) - seq_len + 1):
        seq = telemetry_data.iloc[i:i+seq_len].values
        telemetry_seq.append(seq)

    telemetry_tensor = torch.tensor(np.array(telemetry_seq), dtype=torch.float32)

    # Make predictions
    if model.config.estimate_uncertainty:
        mean, var = predict(model, telemetry_tensor)

        # Convert to numpy
        mean = mean.cpu().numpy()
        var = var.cpu().numpy()
        std = np.sqrt(var)

        # Create DataFrame with predictions
        timestamps = telemetry_data.index[seq_len-1:]
        predictions = pd.DataFrame({
            'Timestamp': timestamps,
            'Predicted_Granulometry_1': mean[:, 0],
            'Predicted_Granulometry_2': mean[:, 1],
            'Std_Granulometry_1': std[:, 0],
            'Std_Granulometry_2': std[:, 1]
        })
    else:
        output = predict(model, telemetry_tensor)

        # Convert to numpy
        output = output.cpu().numpy()

        # Create DataFrame with predictions
        timestamps = telemetry_data.index[seq_len-1:]
        predictions = pd.DataFrame({
            'Timestamp': timestamps,
            'Predicted_Granulometry_1': output[:, 0],
            'Predicted_Granulometry_2': output[:, 1]
        })

    # Save predictions
    predictions.to_excel('predictions.xlsx', index=False)

    print(f"Predictions saved to 'predictions.xlsx'")

if __name__ == '__main__':
    main()
