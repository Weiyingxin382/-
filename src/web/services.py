"""查询服务统一数据版本及连接生命周期。"""

from src.web.database import connect
from src.web import repositories


class QueryService:
    def __init__(self, settings):
        self.settings = settings

    def query(self, operation, dataset_id, *args):
        with connect(self.settings.database) as db:
            db.execute("BEGIN")
            version = repositories.dataset(db, dataset_id)
            value = getattr(repositories, operation)(db, *args)
        return value, version
