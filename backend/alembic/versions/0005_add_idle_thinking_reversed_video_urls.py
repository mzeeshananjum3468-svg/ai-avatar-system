"""add idle_video_reversed_url and thinking_video_reversed_url to avatars

Revision ID: 0005
Revises: 0004
Create Date: 2026-07-31
"""
from alembic import op
import sqlalchemy as sa

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("avatars", sa.Column("idle_video_reversed_url", sa.String(), nullable=True))
    op.add_column("avatars", sa.Column("thinking_video_reversed_url", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("avatars", "thinking_video_reversed_url")
    op.drop_column("avatars", "idle_video_reversed_url")
