import argparse

class BaseParser():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def initialize(self):
        self.parser.add_argument('--device', default='cuda', type=str)
        self.parser.add_argument('--date', default='0410', type=str)
        self.parser.add_argument('--seed', default=7136, type=int)
        self.parser.add_argument('--model', default='Unet', type=str)
        self.parser.add_argument('--model_save_path', default=f"D:\\HIPPO\\")
        self.parser.add_argument('--filename', default='_', type=str)

        self.parser.add_argument('--edge', default=1, type=int)
        self.parser.add_argument('--module', default=1, type=int)

    def parse(self):
        self.initialize()
        self.args = self.parser.parse_args()

        return self.args