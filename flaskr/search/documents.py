from collections import Counter
from dataclasses import dataclass

from flask.helpers import stream_with_context

from .analysis import analyze


@dataclass
class Abstract:
    """job abstract"""
    ID: int
    title: str
    address: str
    url: str
    company: str
    maxSalary: str

    @property
    def fulltext(self):
        return ' '.join([self.title, self.address, self.company])

    def analyze(self):
        self.term_frequencies = Counter(analyze(self.fulltext))

    def term_frequency(self, term):
        return self.term_frequencies.get(term, 0)
