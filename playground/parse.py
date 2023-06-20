import argparse

parser = argparse.ArgumentParser(description='Customized PPO')
parser.add_argument('--bound', type=bool, default=False,
                    help='To use bounded value net or not')

args = parser.parse_args()

print(args.bound)