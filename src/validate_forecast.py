"""Validate the canonical forecast output, defaulting to the validation window."""
import argparse
from experiment_io import EXPERIMENT_DIR, ExperimentOutput
from forecast_experiment import validate_forecast

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--held-out', action='store_true')
    args=parser.parse_args()
    variant='forecast' if args.held_out else 'forecast-development'
    validate_forecast(ExperimentOutput(EXPERIMENT_DIR/variant), held_out=args.held_out)
