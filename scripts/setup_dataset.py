import os
import shutil
from pathlib import Path

def setup_dataset():
    root = Path(__file__).resolve().parent.parent
    data_dir = root / 'data'

    # Clean existing contents in data subdirs
    for split in ['train', 'val', 'test']:
        for category in ['input', 'target']:
            d = data_dir / split / category
            if d.exists():
                for item in d.iterdir():
                    if item.name != '.gitkeep':
                        if item.is_symlink() or item.is_file():
                            item.unlink()
                        elif item.is_dir():
                            shutil.rmtree(item)
            else:
                d.mkdir(parents=True, exist_ok=True)

    # Source dirs
    gt_dir = (root / 'train(2)/train/GT').resolve()
    noisy_dir = (root / 'train(2)/train/NoisyLR').resolve()
    test_src_dir = (root / 'Test_NoisyLR(2)/NoisyLR').resolve()

    if not gt_dir.exists() or not noisy_dir.exists():
        raise FileNotFoundError(f"Source train directory not found: {gt_dir} or {noisy_dir}")

    gt_files = sorted(list(gt_dir.glob('*.npy')))
    noisy_files = sorted(list(noisy_dir.glob('*.npy')))
    test_src_files = sorted(list(test_src_dir.glob('*.npy'))) if test_src_dir.exists() else []

    print(f"Found {len(gt_files)} GT files and {len(noisy_files)} NoisyLR files in train(2).")
    print(f"Found {len(test_src_files)} files in Test_NoisyLR(2).")

    # 2800 for train split, 400 for val split
    train_count = int(len(noisy_files) * 0.875)  # 2800 out of 3200
    train_noisy = noisy_files[:train_count]
    train_gt = gt_files[:train_count]

    val_noisy = noisy_files[train_count:]
    val_gt = gt_files[train_count:]

    # Populate train split
    for noisy_f, gt_f in zip(train_noisy, train_gt):
        os.symlink(noisy_f, data_dir / 'train/input' / noisy_f.name)
        os.symlink(gt_f, data_dir / 'train/target' / gt_f.name)

    # Populate val split
    for noisy_f, gt_f in zip(val_noisy, val_gt):
        os.symlink(noisy_f, data_dir / 'val/input' / noisy_f.name)
        os.symlink(gt_f, data_dir / 'val/target' / gt_f.name)

    # Populate test split
    for test_f in test_src_files:
        os.symlink(test_f, data_dir / 'test/input' / test_f.name)

    print("Dataset setup successfully completed!")
    print(f"  Train inputs:  {len(list((data_dir / 'train/input').glob('*.npy')))}")
    print(f"  Train targets: {len(list((data_dir / 'train/target').glob('*.npy')))}")
    print(f"  Val inputs:    {len(list((data_dir / 'val/input').glob('*.npy')))}")
    print(f"  Val targets:   {len(list((data_dir / 'val/target').glob('*.npy')))}")
    print(f"  Test inputs:   {len(list((data_dir / 'test/input').glob('*.npy')))}")

if __name__ == '__main__':
    setup_dataset()
