from alembic import context
from sqlalchemy import create_engine

from llm_lab.jobs.sqlite import JobTransitionRow

url = context.config.get_main_option("sqlalchemy.url")
if url is None:
    raise ValueError("the Alembic configuration has no sqlalchemy.url option")

engine = create_engine(url)
with engine.connect() as connection:
    context.configure(connection=connection, target_metadata=JobTransitionRow.metadata, render_as_batch=True)
    with context.begin_transaction():
        context.run_migrations()
engine.dispose()
