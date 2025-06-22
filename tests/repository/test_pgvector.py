import math
from typing import List

from pytest import fixture, mark
from testcontainers.postgres import PostgresContainer

from app.repository.pgvector import InsertVector, PgVectorRepository, VectorRecord


@fixture(scope="function")
def pgvector():
    container = PostgresContainer("pgvector/pgvector:pg17")
    container.start()
    connection_string = container.get_connection_url(driver=None)
    yield connection_string
    container.stop()


@fixture(scope="module")
def dummy_data() -> List[InsertVector]:
    zodiac = [
        (
            "牡羊座",
            "3/21-4/19. 主要な星はハマル、シェラタン、メサルシム、アルフェラツ、アダーラで構成される。アルファ星のハマルは羊の頭を表し、ベータ星のシェラタンは羊の角を意味する。",
        ),
        (
            "金牛座",
            "4/20-5/20. 主要な星はアルデバラン、エルナト、ゼータ・タウリ、アルキオネ、マイア、エレクトラ、タイゲタ、メロペ、ケラエノ、ステロペで構成される。プレアデス星団とヒアデス星団を含む。",
        ),
        (
            "双子座",
            "5/21-6/21. 主要な星はカストル、ポルックス、アルヘナ、ワサト、メブスタ、プロプス、テジャット・プリオル、テジャット・ポステリオルで構成される。カストルとポルックスは双子の兄弟を表す。",
        ),
        (
            "蟹座",
            "6/22-7/22. 主要な星はアクベンス、アル・タルフ、イオタ・カンクリ、デルタ・カンクリで構成される。中央にはプレセペ星団（蜂の巣星団）がある。比較的暗い星座で目立たない。",
        ),
        (
            "獅子座",
            "7/23-8/22. 主要な星はレグルス、デネボラ、アルギエバ、ゾスマ、アダフェラ、ラス・エラセド・アウストラリス、ラス・エラセド・ボレアリス、スバルで構成される。レグルスは王の星として知られる。",
        ),
        (
            "乙女座",
            "8/23-9/22. 主要な星はスピカ、ガンマ・ヴィルギニス、イプシロン・ヴィルギニス、デルタ・ヴィルギニス、ベータ・ヴィルギニス、ザヴィヤヴァ、ミンエラウヴァで構成される。スピカは麦の穂を表す。",
        ),
        (
            "天秤座",
            "9/23-10/23. 主要な星はズベン・エル・ゲヌビ、ズベン・エル・シャマリ、ズベン・エル・アクラブ、ガンマ・リブラエ、シグマ・リブラエで構成される。古代は蠍座の一部とされていた。",
        ),
        (
            "蠍座",
            "10/24-11/22. 主要な星はアンタレス、シャウラ、サルガス、イータ・スコルピイ、ゼータ・スコルピイ、ミュー・スコルピイ、イプシロン・スコルピイで構成される。アンタレスは火星のライバルという意味。",
        ),
        (
            "射手座",
            "11/23-12/21. 主要な星はサギッタリウス、ヌンキ、アスケラ、カウス・アウストラリス、カウス・メディウス、カウス・ボレアリス、アルバルダ、アルナスルで構成される。銀河系の中心方向にある。",
        ),
        (
            "山羊座",
            "12/22-1/19. 主要な星はアルゲディ、ダビー、ナシラ、デネブ・アルゲディ、バテン・アルゲディ、オメガ・カプリコルニで構成される。比較的暗い星で構成される山羊の形を表している。",
        ),
        (
            "水瓶座",
            "1/20-2/18. 主要な星はサダルメリク、サダルスウド、スカト、サダクビア、アルバリ、アンカ、シタラで構成される。水を注ぐ人の姿を表し、多くの球状星団がある。",
        ),
        (
            "魚座",
            "2/19-3/20. 主要な星はアル・リシャ、フマル・サマカ、トルクラリウム・セプテントリオナレ、ガンマ・ピスキウム、ベータ・ピスキウムで構成される。二匹の魚がリボンで結ばれた形を表している。",
        ),
    ]
    data = []
    for i in range(12):
        name, discription = zodiac[i]
        angle = i * 30
        radian = math.radians(angle)
        sin_value = math.sin(radian)
        cos_value = math.cos(radian)
        data.append(
            InsertVector(
                category="前半" if i < 6 else "後半",
                title=f"星座：{name}",
                text=discription,
                embedding=[cos_value, sin_value],
            )
        )
    return data


@mark.ut
def test_create_table(pgvector: str):
    # given
    repo = PgVectorRepository(pgvector)
    # when
    repo.create_table(vector_dim=2)
    # then
    with repo.conn.cursor() as cursor:
        cursor.execute("SELECT to_regclass('embeddings');")
        result = cursor.fetchone()
        assert result[0] == "embeddings"

        cursor.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name='embeddings';"
        )
        columns = [row[0] for row in cursor.fetchall()]
        expected_columns = [
            "id",
            "category",
            "title",
            "text",
            "embedding",
            "created_at",
        ]
        assert set(columns) == set(expected_columns)


@mark.ut
@mark.parametrize(
    "embedding, query, category, expected_title",
    [
        ([1.0, 0.0], "関係ないクエリ", "前半", "星座：牡羊座"),
        ([1.0, 0.0], "関係ないクエリ", None, "星座：牡羊座"),
        ([0.0, 0.0], "獅子座", "前半", "星座：獅子座"),
        ([0.0, 0.0], "獅子座", None, "星座：獅子座"),
        (
            [0.0, 0.0],
            "アンタレス/シャウラ/サルガス/イータ・スコルピイ/ゼータ・スコルピイ/ミュー・スコルピイ/イプシロン・スコルピイ",
            "後半",
            "星座：蠍座",
        ),
        (
            [0.0, 0.0],
            "アンタレス/シャウラ/サルガス/イータ・スコルピイ/ゼータ・スコルピイ/ミュー・スコルピイ/イプシロン・スコルピイ",
            None,
            "星座：蠍座",
        ),
    ],
)
def test_hybrid_search(
    pgvector: str,
    dummy_data: List[InsertVector],
    embedding: List[float],
    query: str,
    category: str,
    expected_title: str,
):
    # given
    repo = PgVectorRepository(pgvector)
    repo.create_table(vector_dim=2)
    repo.copy(dummy_data)

    # when
    results: List[VectorRecord] = repo.hybrid_search(
        embedding=embedding, query=query, category=category, limit=5
    )

    # then
    assert len(results) <= 5
    print(results)
    assert results[0].title.startswith(expected_title)


@mark.ut
def test_insert(pgvector: str, dummy_data: List[InsertVector]):
    # given
    repo = PgVectorRepository(pgvector)
    repo.create_table(vector_dim=2)

    # when
    repo.copy(dummy_data)

    # then
    with repo.conn.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM embeddings;")
        count = cursor.fetchone()[0]
        assert count == len(dummy_data)
