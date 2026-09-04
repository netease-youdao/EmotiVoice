import os

from fastapi import Request, Response, APIRouter
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="speakers/templates")


@router.get("/")
def read_root(request: Request):
    content = templates.TemplateResponse("index.html", {"request": request})
    return content


@router.get("/speakers")
def read_speakers(request: Request):
    # from openaiapi import config
    # return {"speakers": config.speakers}
    root_dir = os.path.dirname(os.path.abspath("__file__"))
    with open(f"{root_dir}/data/youdao/text/speaker2", encoding = "UTF-8") as f:
        speakers = [t.strip() for t in f.readlines()]
    return {"speakers": speakers}