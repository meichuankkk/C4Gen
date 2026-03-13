from __future__ import annotations

from dataclasses import dataclass

from .helpers import format_name
from .subpkg.adapter import send


@dataclass
class Person:
    name: str

    def greet(self) -> str:
        formatted = format_name(self.name)
        return f"Hello, {formatted}"


@dataclass
class Employee(Person):
    role: str

    def greet(self) -> str:
        base = super().greet()
        return f"{base} ({self.role})"

    def notify(self) -> str:
        return send(self.greet())


def make_person(name: str) -> Person:
    return Person(name=name)


def make_employee(name: str, role: str) -> Employee:
    return Employee(name=name, role=role)

