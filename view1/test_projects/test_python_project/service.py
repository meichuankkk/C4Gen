from .helpers import default_role
from .models import Employee, make_employee


def run(user_name: str) -> str:
    role = default_role()
    emp = make_employee(user_name, role)
    return emp.notify()


def as_employee(user_name: str) -> Employee:
    return make_employee(user_name, default_role())

