from typing import Optional

import pytest
from pydantic import field_validator
from pydantic.error_wrappers import ValidationError
from sqlmodel import SQLModel, create_engine, Session, Field


def test_model_copy(clear_sqlmodel):
    """Test validation of implicit and explict None values.

    # For consistency with pydantic, validators are not to be called on
    # arguments that are not explicitly provided.

    https://github.com/tiangolo/sqlmodel/issues/230
    https://github.com/samuelcolvin/pydantic/issues/1223

    """

    class Hero(SQLModel, table=True):
        id: Optional[int] = Field(default=None, primary_key=True)
        name: str
        secret_name: str
        age: Optional[int] = None

    hero = Hero(name="Deadpond", secret_name="Dive Wilson", age=25)

    engine = create_engine("sqlite://")

    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        session.add(hero)
        session.commit()
        session.refresh(hero)

    model_copy = hero.model_copy(update={"name": "Deadpond Copy"})

    assert model_copy.name == "Deadpond Copy" and \
           model_copy.secret_name == "Dive Wilson" and \
           model_copy.age == 25

    db_hero = session.get(Hero, hero.id)

    db_copy = db_hero.model_copy(update={"name": "Deadpond Copy"})

    assert db_copy.name == "Deadpond Copy" and \
           db_copy.secret_name == "Dive Wilson" and \
           db_copy.age == 25
