"""create job transitions"""

import sqlalchemy as sa
from alembic import op

revision: str = "9395d98d9cb4"
down_revision: str | None = None
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None


def upgrade() -> None:
    op.create_table(
        "job_transitions",
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("condition", sa.String(), nullable=False),
        sa.Column("recorded_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.Column("record", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("sequence"),
        sqlite_autoincrement=True,
    )
    with op.batch_alter_table("job_transitions", schema=None) as batch_op:
        batch_op.create_index("job_transitions_by_name", ["name", "sequence"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("job_transitions", schema=None) as batch_op:
        batch_op.drop_index("job_transitions_by_name")

    op.drop_table("job_transitions")
