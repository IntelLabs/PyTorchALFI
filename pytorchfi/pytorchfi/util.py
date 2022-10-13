"""
pytorchfi.util contains utility functions to help the user generate fault
injections and determine their impact.
"""

import time

import torch
import torch.nn as nn
from ..pytorchfi import core

class Map_Dict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __repr__(self):
        return '<DictX ' + dict.__repr__(self) + '>'

class util(core.fault_injection):
    def compare_golden(self, input_data):
        softmax = nn.Softmax(dim=1)

        model = self.get_original_model()
        golden_output = model(input_data)
        golden_output_softmax = softmax(golden_output)
        golden = list(torch.argmax(golden_output_softmax, dim=1))

        corrupted_model = self.get_corrupted_model()
        corrupted_output = corrupted_model(input_data)
        corrupted_output_softmax = softmax(corrupted_output)
        corrupted = list(torch.argmax(corrupted_output_softmax, dim=1))

        return [golden, corrupted]

    def time_model(self, model, input_data, iterations=100):
        start_time = time.time()
        for _ in range(iterations):
            model(input_data)
        end_time = time.time()
        return (end_time - start_time) / iterations

class CircularBuffer(object):

    def __init__(self, max_size=10):
        """Initialize the CircularBuffer with a max_size if set, otherwise
        max_size will elementsdefault to 10"""        
        self.head = 0
        self.tail = 0
        self.max_size = max_size
        self.buffer = [None] * self.max_size

    def __str__(self):
        """Return a formatted string representation of this CircularBuffer."""
        items = ['{!r}'.format(item) for item in self.buffer]
        return '[' + ', '.join(items) + ']'

    def size(self):
        """Return the size of the CircularBuffer
        Runtime: O(1) Space: O(1)"""
        if self.tail >= self.head:
            return self.tail - self.head
        return self.max_size - self.head - self.tail

    def is_empty(self):
        """Return True if the head of the CircularBuffer is equal to the tail,
        otherwise return False
        Runtime: O(1) Space: O(1)"""
        return self.tail == 0

    def is_full(self):
        """Return True if the tail of the CircularBuffer is one before the head,
        otherwise return False
        Runtime: O(1) Space: O(1)"""
        return self.tail == self.max_size

    def enqueue(self, item):
        """Insert an item at the back of the CircularBuffer
        Runtime: O(1) Space: O(1)"""
        if self.is_full():
            raise OverflowError(
                "CircularBuffer is full, unable to enqueue item")
        self.buffer[self.tail] = item
        self.tail = (self.tail + 1)

    def front(self):
        """Return the item at the front of the CircularBuffer
        Runtime: O(1) Space: O(1)"""
        if self.is_empty():
            raise IndexError("CircularBuffer is empty, unable to dequeue")
        item = self.buffer[self.head]
        self.head = (self.head + 1) % self.max_size
        return item

    def dequeue(self):
        """Return the item at the front of the Circular Buffer and remove it
        Runtime: O(1) Space: O(1)"""
        if self.is_empty():
            raise IndexError("CircularBuffer is empty, unable to dequeue")
        item = self.buffer[self.head]
        self.buffer[self.head] = None
        self.head = (self.head + 1) % self.max_size
        return item
    
    def get_content_list(self):
        return self.buffer

def get_savedBounds_minmax(filename):
    f = open(filename, "r")
    bounds = []
    if f.mode == 'r':
        contents = f.read().splitlines()
        bounds = [u.split(',') for u in contents]
    f.close()

    bounds = [[float(n[0]), float(n[1])] for n in bounds] #make numeric

    return bounds