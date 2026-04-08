# server/app.py

from openenv.core.env_server.http_server import create_app
from env.environment import EmailTriageEnv
from env.models import EmailTriageAction, EmailTriageObservation


def env_factory(**kwargs):
    """
    Factory to create a new EmailTriageEnv instance.
    Accepts optional 'task_name' keyword to start a specific task.
    Defaults to 'task1_classify'.
    """
    task_name = kwargs.get("task_name", "task1_classify")
    return EmailTriageEnv(task_name=task_name)


# Create the FastAPI app using OpenEnv's auto-generated endpoints
app = create_app(
    env_factory,
    action_cls=EmailTriageAction,
    observation_cls=EmailTriageObservation
)


def main():
    """
    Main entry point to run the server with Uvicorn.
    Host 0.0.0.0 makes it reachable from Docker/remote.
    """
    import uvicorn
    uvicorn.run(
        "server.app:app",  # module:app
        host="0.0.0.0",
        port=8000,
        reload=True,       # auto-reload on code changes for development
        log_level="info"
    )


if __name__ == "__main__":
    main()