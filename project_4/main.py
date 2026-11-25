import argparse
import torch
from pathlib import Path

from proposals import GenerateAllProposals
from evaluate_proposals import EvaluateProposals

# to use best proposals:                        python main.py --quality
# to use worse proposals:                       python main.py
# if you wish to generate other proposals:      python main.py --mode (choose mode) --size (optionally)

# note that selective search is deterministic, running the same mode/size combo twice will give identical proposals
   

                                        
def main():
    # run configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['fast', 'quality'], default='fast') # choose selective search mode
    parser.add_argument('--size', nargs=2, type=int, default=None) #optional resize images (uniform size=decreased runtime)
    parser.add_argument('--regenerate', action='store_true')
    args = parser.parse_args()

    #info for generating output file name
    size_name = 'original' if args.size is None else f'{args.size[0]}x{args.size[1]}'
    output_path = f'ss_{args.mode}_{size_name}.pt'

    size = tuple(args.size) if args.size else None

    # reuse existing proposals if available
    if Path(output_path).exists() and not args.regenerate:
        print(f'Loading existing proposals: {output_path}')
        results = torch.load(output_path)
    else:
        print(f'Generating proposals: {output_path}')
        results = GenerateAllProposals(mode=args.mode, size=size)

        # save as
        torch.save(results, output_path)

    #evaluate
    counts = [500, 750, 1000, 1250, 1500, 2000, 5000] # counts are how many proposals per image we want to evaluate
    #this range is something we can tune, choosing what works based on results

    # tuning found ideal counts: mode=fast counts=2000, mode=quality counts=5000
    EvaluateProposals(results, counts)


if __name__ == '__main__':
    main()