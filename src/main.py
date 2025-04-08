import argparse
import yaml

from src.run import run_experiment

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    run_experiment(config)

if __name__ == "__main__":
    main()