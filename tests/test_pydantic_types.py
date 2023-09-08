from pydantic import EmailStr, HttpUrl, ImportString, NameEmail
from sqlmodel import Field, SQLModel, create_engine


def test_pydantic_types(clear_sqlmodel, caplog):
    class Hero(SQLModel, table=True):
        integer_primary_key: int = Field(
            primary_key=True,
        )
        http: HttpUrl = Field(max_length=250)
        email: EmailStr
        name_email: NameEmail = Field(max_length=50)
        import_string: ImportString = Field(max_length=200, min_length=100)

    engine = create_engine("sqlite://", echo=True)
    SQLModel.metadata.create_all(engine)

    create_table_log = [
        message for message in caplog.messages if "CREATE TABLE hero" in message
    ][0]
    assert "http VARCHAR(250) NOT NULL," in create_table_log
    assert "email VARCHAR NOT NULL," in create_table_log
    assert "name_email VARCHAR(50) NOT NULL," in create_table_log
    assert "import_string VARCHAR(200) NOT NULL," in create_table_log
