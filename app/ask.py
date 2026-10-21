"""第1回: 素のLLMに1回だけ質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする
    uv run python -m app.ask --temperature 0.0 "LoRAとは何ですか？"  # temperature を指定

ツールも会話履歴も持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
from google.genai import errors, types

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def ask(question: str, temperature: float | None = None) -> str:
    """LLMに1回だけ質問し、入力・出力トークン数を表示して回答を返す。

    Args:
        question: LLMへの質問文。
        temperature: 生成の temperature。None ならモデルの既定値を使う。

    Returns:
        LLMの回答テキスト。

    Raises:
        google.genai.errors.APIError: API呼び出しに失敗した場合。
    """
    client = genai.Client(api_key=get_api_key())
    config = None if temperature is None else types.GenerateContentConfig(temperature=temperature)
    response = client.models.generate_content(model=MODEL, contents=question, config=config)

    # 第1回 演習: response.usage_metadata から入力・出力トークン数を表示する
    usage = response.usage_metadata
    if usage is not None:
        print(
            f"[tokens] 入力: {usage.prompt_token_count} / 出力: {usage.candidates_token_count}"
            f" / 思考: {usage.thoughts_token_count or 0} / 合計: {usage.total_token_count}\n"
        )

    return response.text


def main() -> None:
    """コマンドライン引数を読み、LLMに質問して結果を表示する。"""
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--temperature", type=float, default=None, help="生成の temperature（省略時はモデルの既定値）")
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    if args.temperature is not None:
        print(f"[temperature] {args.temperature}")
    print(f"[質問] {args.question}\n")
    try:
        print(ask(args.question, args.temperature))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
