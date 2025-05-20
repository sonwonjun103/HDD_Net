from options.BaseOption import BaseParser

class TrainParser(BaseParser):
    def initialize(self):
        BaseParser.initialize(self)
        self.parser.add_argument('--lr', default=0.0001, type=float)
        self.parser.add_argument('--batch_size', default=4, type=int)
        self.parser.add_argument('--shuffle', default=True, type=bool)
        self.parser.add_argument('--epochs', default=150, type=int)

        self.parser.add_argument('--min_window', default=-20, type=int)
        self.parser.add_argument('--max_window', default=100, type=int)

        self.parser.add_argument('--gaussian_filter', default=1, type=int)
        self.parser.add_argument('--filter_threshold', default=0.4, type=float)
        self.parser.add_argument('--bone_threshold', default=0.8, type=float)

        self.parser.add_argument('--crop_size', default=128, type=int)
        self.parser.add_argument('--depth_crop_size', default=96, type=int)
        self.parser.add_argument('--is_train', default=True, type=bool)