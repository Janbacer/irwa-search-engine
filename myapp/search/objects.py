"""Data objects of the web application: a product of the dataset (``Document``) and a result record (``ResultItem``)."""
from pydantic import BaseModel, field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime
import re


class Document(BaseModel):
    """One product of the dataset, with its prices, discount and rating parsed into numbers."""
    _id: str
    pid: str
    title: str
    description: Optional[str] = None
    brand: Optional[str] = None
    category: Optional[str] = None
    sub_category: Optional[str] = None
    product_details: Optional[Dict[str, Any]] = None
    seller: Optional[str] = None
    out_of_stock: bool = False
    selling_price: Optional[float] = None
    discount: Optional[float] = None
    actual_price: Optional[float] = None
    average_rating: Optional[float] = None
    url: Optional[str] = None
    images: Optional[List[str]] = None
    crawled_at: Optional[int] = None     # milliseconds since 1970 (date of the crawl)

    def to_json(self):
        """The product as a JSON string."""
        return self.model_dump_json()

    # --- Validators ---

    @field_validator("selling_price", "actual_price", mode="before")
    def parse_price(cls, v):
        """'2,999' -> 2999.0; empty -> None."""
        if v is None:
            return None
        if isinstance(v, str):
            v = v.strip().replace(",", "")
            if v == "":
                return None
            try:
                return float(v)
            except ValueError:
                return None
        return v

    @field_validator("average_rating", mode="before")
    def parse_rating(cls, v):
        """'3.9' -> 3.9; empty -> None (a missing rating is not a rating of 0)."""
        if v is None:
            return None
        if isinstance(v, str):
            v = v.strip()
            if v == "":
                return None
            try:
                return float(v)
            except ValueError:
                return None
        return v

    @field_validator("discount", mode="before")
    def parse_discount(cls, v):
        """'69% off' -> 69.0; empty -> None."""
        if v is None:
            return None
        if isinstance(v, str):
            match = re.search(r"(\d+(?:\.\d+)?)", v.replace(",", ""))
            if match:
                return float(match.group(1))
            return None
        return v

    @field_validator("product_details", mode="before")
    def normalize_product_details(cls, v):
        """[{'Fabric': 'Cotton'}, {'Color': 'Blue'}] -> {'Fabric': 'Cotton', 'Color': 'Blue'}."""
        if isinstance(v, list):
            merged = {}
            for item in v:
                if isinstance(item, dict):
                    merged.update(item)
            return merged
        return v

    def __str__(self) -> str:
        return self.model_dump_json(indent=2)


class ResultItem(BaseModel):
    """One product of a results page: the fields shown to the user plus its rank and score."""
    pid: str
    rank: int
    score: float
    title: str
    description: Optional[str] = None
    brand: Optional[str] = None
    sub_category: Optional[str] = None
    selling_price: Optional[float] = None
    actual_price: Optional[float] = None
    discount: Optional[float] = None
    average_rating: Optional[float] = None
    in_stock: bool = True
    url: Optional[str] = None            # link to the product on the original website
    image: Optional[str] = None
    details: Dict[str, str] = {}         # product_details as a flat dictionary

    def __str__(self) -> str:
        return self.model_dump_json(indent=2)

    def to_json(self):
        """The result as a JSON string."""
        return self.model_dump_json()
