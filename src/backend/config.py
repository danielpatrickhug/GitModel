from pydantic import BaseModel, BaseSettings


class MessageTreeManagerConfiguration(BaseModel):
    pass


class Settings(BaseSettings):
    pass


settings = Settings()
