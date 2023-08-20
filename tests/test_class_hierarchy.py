import datetime
import sys

import pytest
from pydantic import AnyUrl, UrlConstraints
from sqlmodel import (
    BigInteger,
    Column,
    DateTime,
    Field,
    Integer,
    SQLModel,
    String,
    create_engine,
)
from typing_extensions import Annotated

MoveSharedUrl = Annotated[
    AnyUrl, UrlConstraints(max_length=512, allowed_schemes=["smb", "ftp", "file"])
]


@pytest.mark.skipif(sys.version_info < (3, 10))
def test_field_resuse():
    class BasicFileLog(SQLModel):
        resourceID: int = Field(
            sa_column=lambda: Column(Integer, index=True), description="""   """
        )
        transportID: Annotated[int | None, Field(description=" for ")] = None
        fileName: str = Field(
            sa_column=lambda: Column(String, index=True), description=""" """
        )
        fileSize: int | None = Field(
            sa_column=lambda: Column(BigInteger), ge=0, description=""" """
        )
        beginTime: datetime.datetime | None = Field(
            sa_column=lambda: Column(
                DateTime(timezone=True),
                index=True,
            ),
            description="",
        )

    class SendFileLog(BasicFileLog, table=True):
        id: int | None = Field(
            sa_column=Column(Integer, primary_key=True, autoincrement=True),
            description="""   """,
        )
        sendUser: str
        dstUrl: MoveSharedUrl | None

    class RecvFileLog(BasicFileLog, table=True):
        id: int | None = Field(
            sa_column=Column(Integer, primary_key=True, autoincrement=True),
            description="""   """,
        )
        recvUser: str

    sqlite_file_name = "database.db"
    sqlite_url = f"sqlite:///{sqlite_file_name}"

    engine = create_engine(sqlite_url, echo=True)
    SQLModel.metadata.drop_all(engine)
    SQLModel.metadata.create_all(engine)
    SendFileLog(
        sendUser="j",
        resourceID=1,
        fileName="a.txt",
        fileSize=3234,
        beginTime=datetime.datetime.now(),
    )
    RecvFileLog(
        sendUser="j",
        resourceID=1,
        fileName="a.txt",
        fileSize=3234,
        beginTime=datetime.datetime.now(),
    )
