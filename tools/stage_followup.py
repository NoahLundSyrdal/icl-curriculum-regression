"""Stage the new follow-up source without altering the recorded cluster experiment."""
import argparse,shutil,subprocess
from pathlib import Path

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--destination',type=Path,required=True);a=parser.parse_args()
    repo=Path(__file__).resolve().parents[1];source=repo/'experiments/followup_20261013'
    destination=a.destination.resolve()
    if destination==source or source in destination.parents:raise ValueError('Stage outside archived source')
    if destination.exists() and any(destination.iterdir()):raise FileExistsError('Destination is not empty')
    destination.mkdir(parents=True,exist_ok=True)
    for file in source.iterdir():
        if file.is_file() and file.suffix in ('.py','.slurm','.json','.md'):shutil.copyfile(file,destination/file.name)
    (destination/'submissions').mkdir()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    (destination/'submissions/repository_commit.txt').write_text(commit+'\n')
    print(destination)
if __name__=='__main__':main()
