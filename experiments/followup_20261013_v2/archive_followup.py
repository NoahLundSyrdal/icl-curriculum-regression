"""Archive exact run text after successful summary; never include weights or credentials."""
import hashlib,json,zipfile
from pathlib import Path
import followup_plan as plan

def main():
    plan.verify_sources();plan.load_selection()
    summary=json.loads(Path('results/summary.json').read_text())
    if summary['kind']!=plan.KIND or summary['freeze_sha256']!=plan.digest(plan.FREEZE):raise ValueError('Summary incompatible')
    destination=Path('results/followup_20261013_v2.zip')
    if destination.exists():raise FileExistsError('Archive exists; inspect instead of overwrite')
    files={Path(name) for name in plan.FILES}
    for root in (Path('submissions'),Path('logs'),Path('results')):
        for file in root.rglob('*'):
            if file.is_file() and file.suffix in ('.json','.csv','.txt','.out','.err','.md'):
                files.add(file)
    temporary=destination.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for file in sorted(files):z.write(file,file.as_posix())
    temporary.replace(destination)
    report=dict(kind=plan.KIND,archive=destination.as_posix(),files=len(files),bytes=destination.stat().st_size,
        sha256=plan.digest(destination),created_utc=plan.now(),freeze_sha256=plan.digest(plan.FREEZE),
        note='Includes complete raw metrics, train logs, source and submission evidence; model weights/schedules stay on cluster. Original summary audited retained binaries.')
    plan.atomic_json('results/archive_manifest.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
