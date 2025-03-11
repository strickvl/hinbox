"""add_article_table

Revision ID: 73ae7f5dc49e
Revises: 72d2b0000b8a
Create Date: 2025-02-25 07:58:02.929772

"""

from typing import Sequence, Union
from uuid import uuid4

import sqlalchemy as sa
import sqlmodel
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "73ae7f5dc49e"
down_revision: Union[str, None] = "72d2b0000b8a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create article table
    op.create_table(
        "article",
        sa.Column("id", sqlmodel.sql.sqltypes.GUID(), nullable=False, default=uuid4),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("url", sa.String(), nullable=False),
        sa.Column("published_date", sa.DateTime(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("scrape_timestamp", sa.DateTime(), nullable=False),
        sa.Column("content_scrape_timestamp", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("processed", sa.Boolean(), nullable=False, default=False),
        sa.Column("processing_timestamp", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create indexes
    op.create_index(op.f("ix_article_title"), "article", ["title"], unique=False)
    op.create_index(op.f("ix_article_url"), "article", ["url"], unique=True)
    op.create_index(
        op.f("ix_article_published_date"), "article", ["published_date"], unique=False
    )


def downgrade() -> None:
    # Drop article table
    op.drop_index(op.f("ix_article_published_date"), table_name="article")
    op.drop_index(op.f("ix_article_url"), table_name="article")
    op.drop_index(op.f("ix_article_title"), table_name="article")
    op.drop_table("article")
