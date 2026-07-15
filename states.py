# So whenever we want to create a Graph, the first thing we have to create a state 
# state is used maintain the flow of data 
# to create a state we have three methods


import os
from typing import TypedDict

class State(TypedDict):
    # here we write or make the skeleton what we want to create 
    # first method 
    topic : str
    summary : str
    score : int

# second method is pydentic approach 
# it was mostly used for type checking and data validation at run-time

from pydantic import BaseModel, field_validaor

class State(BaseModel):
    topic : str
    score : int
    summary : str = "no summary "

    @field_validaor
    def score_validator(csl,v):
        if v < 0:
            raise ValueError("Score Must be Positive")


# python data classes 
# standard python data classes but it is used vary rerelty

from dataclasses import dataclass, field 

@dataclass
class state:
    topic : str = ""
    summary : str = ""
    score : list = field(default_factory=list)