"""
This is a effective random uniform sampling module known as
Reservoir Sampling which uniformly samples a subset of k elements from
a larger 'reservoir' of size N where N may be unknown or very large.


Examples:
---------
    The command line tool itself takes in either a stream or a file.
    The following examples samples 100 elements a streaming twitter
    api and a text file respectably.

        $ twitter-streaming-api | python sample.py --stream 100
        $ python sample.py --file tweets.txt 100
        $ python sample.py -h

Attributes:
-----------
    UniformSampler (Object): This module contains a UniformClass object with
        an `feed` method that takes in the ith element and `stream_sample` that
        processes any iterable and returns the sampled elements.

Author: Jason Liu
Date: May 24th, 2014

"""
# from https://github.com/jxnl/python-reservoir
from __future__ import print_function

import sys
from argparse import ArgumentParser
from random import randint


class UniformSampler(object):

    """Uniformly sample k elements from an iterable.

    Attributes:
        sample (list): Reservoir of random samples.
        counter (int): Saves the total number of values seen.
        _max (int): The max number of samples allowed in memory.

    """

    def __init__(self, size=10):
        self.sample = list()
        self.counter = int()
        self._max = size

    def feed(self, item):
        """Feed new values into sampler and insert if selected.

        Args:
            item: Item you wish you sample next.

        Returns:
            The resulting reservoir after sampling.

        """
        self.counter += 1
        switch = randint(0, self.counter)
        if len(self.sample) < self._max:
            self.sample.append(item)
        elif switch < self._max:
            self.sample[switch] = item

        return self.sample

    def stream_sample(self, stream):
        """Sample k elements from an iterable! Duh.

        Args:
            stream: Any iterable object to sample from.

        Returns:
            Rhe list of elements sampled from the iterable

        """
        for item in stream:
            self.feed(item)

        return self.sample

    def __repr__(self):
        return str(self.sample)

    def __str__(self):
        output = str()
        for item in self.sample:
            output += "{item}\n".format(item=item.strip())  # strip?
        return output


def gen_parser():
    """Generate the CLI argparse object"""
    desc = """Random sampling is often applied to very large datasets and
              in particular to data streams. In this case, the random sample
              has to be generated in one pass over an potentially unknown 
              population.
              """
    parser = ArgumentParser(description=desc)
    group = parser.add_mutually_exclusive_group()
    parser.add_argument(
        "size", help="The number of elements you wish you sample", type=int
    )
    group.add_argument("-f", "--file", default=False, type=str)
    group.add_argument(
        "-s",
        "--stream",
        default=False,
        action="store_true",
        help="For use with pipes and redirects",
    )
    return parser


def main():
    parser = gen_parser()
    args = parser.parse_args()
    sampler = UniformSampler(size=args.size)

    if args.file:
        with open(args.file) as res:
            sampler.stream_sample(res)
            print(sampler)

    if args.stream:
        for line in sys.stdin:
            sampler.feed(line.strip())
        print(sampler)


if __name__ == "__main__":
    sys.exit(main())
