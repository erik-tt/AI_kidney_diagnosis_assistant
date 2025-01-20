from argparse import ArgumentParser
from monai.data import DataLoader
from utils.file_reader import FileReader
from utils.create_dataset import create_dataset

def main(params):
    train_dataset, test_dataset = create_dataset(params.data)

    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers)
    val_dataloader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False, num_workers=params.num_workers)   

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data", nargs='+', default=["drsprg/post"], help="List of data directories")
    parser.add_argument("--model", default="UNet")
    parser.add_argument("--batch_size", default=4)
    parser.add_argument("--num_workers", default=0)
    parser.add_argument("--num_epochs", default=10)
    parser.add_argument("--lr", default=0.001)

    args = parser.parse_args()

    main(args)

