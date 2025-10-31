from faker import Faker
import pandas as pd


def make_df(n_rows: int = 10000, ru_share: float = 0.5) -> pd.DataFrame:
    n_rows = int(n_rows)
    n_ru = int(n_rows * ru_share)
    n_en = n_rows - n_ru

    faker_ru = Faker("ru_RU")
    faker_en = Faker("en_US")

    rows: list[dict] = []

    for _ in range(n_ru):
        rows.append(
            {
                "full_name": faker_ru.name(),
                "email": faker_ru.email(),
                "phone": faker_ru.phone_number(),
                "birth_date": faker_ru.date_of_birth(minimum_age=18, maximum_age=90),
                "address": faker_ru.address(),
                "passport_id": faker_ru.bothify(text="#### ######"),
                "bank": faker_ru.credit_card_number(),
            }
        )
    for _ in range(n_en):
        rows.append(
            {
                "full_name": faker_en.name(),
                "email": faker_en.email(),
                "phone": faker_en.phone_number(),
                "birth_date": faker_en.date_of_birth(minimum_age=18, maximum_age=90),
                "address": faker_en.address(),
                "passport_id": faker_en.bothify(text="#### ######"),
                "bank": faker_en.credit_card_number(),
            }
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = make_df(1000, ru_share=0.5)
    print(df)
