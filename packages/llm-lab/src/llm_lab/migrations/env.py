from alembic import context
from sqlalchemy import Engine

from llm_lab.jobs.tables import JobTransitionRow

match context.config.attributes.get("engine"):
    case Engine() as engine:
        with engine.connect() as connection:
            context.configure(connection=connection, target_metadata=JobTransitionRow.metadata, render_as_batch=True)
            with context.begin_transaction():
                context.run_migrations()
    case _:
        raise TypeError("the Alembic configuration has no engine attribute")
