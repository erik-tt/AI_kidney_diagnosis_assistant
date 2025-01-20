from argparse import ArgumentParser

def main(params):
    print(params)

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--model", default="UNet")
    parser.add_argument("--batch_size", default=4)
    parser.add_argument("--num_workers", default=0)
    parser.add_argument("--num_epochs", default=10)
    parser.add_argument("--lr", default=0.001)

    args = parser.parse_args()

    main(args)

