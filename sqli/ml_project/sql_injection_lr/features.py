"""TF-IDF and handcrafted SQL feature extraction."""

from __future__ import annotations

import math
import re
from collections import Counter

from .config import FeatureConfig
from .preprocessing import clean_text, make_ngrams, tokenize


SQL_KEYWORDS = (
    "select",
    "union",
    "drop",
    "insert",
    "update",
    "delete",
    "where",
    "from",
    "or",
    "and",
    "exec",
    "alter",
    "create",
    "sleep",
    "benchmark",
    "information_schema",
    "xp_cmdshell",
)


class TfidfVectorizerScratch:
    """Sparse unigram/bigram TF-IDF vectorizer implemented from scratch."""

    def __init__(self, config: FeatureConfig | None = None):
        self.config = config or FeatureConfig()
        self.vocabulary_: dict[str, int] = {}
        self.idf_: list[float] = []
        self.feature_names_: list[str] = []
        self.n_docs_: int = 0

    def _terms(self, text: str) -> list[str]:
        tokens = tokenize(text)
        return make_ngrams(tokens, self.config.ngram_min, self.config.ngram_max)

    def fit(self, texts: list[str]) -> "TfidfVectorizerScratch":
        doc_freq: Counter[str] = Counter()
        term_freq: Counter[str] = Counter()
        self.n_docs_ = len(texts)
        for text in texts:
            terms = self._terms(text)
            term_freq.update(terms)
            doc_freq.update(set(terms))

        max_df = max(1, int(self.config.max_df_ratio * max(1, self.n_docs_)))
        candidates = [
            term
            for term, df in doc_freq.items()
            if df >= self.config.min_df and df <= max_df
        ]
        candidates.sort(key=lambda term: (-term_freq[term], term))
        selected = candidates[: self.config.max_features]
        selected.sort()

        self.feature_names_ = selected
        self.vocabulary_ = {term: index for index, term in enumerate(selected)}
        self.idf_ = [
            math.log((1 + self.n_docs_) / (1 + doc_freq[term])) + 1.0
            for term in selected
        ]
        return self

    def transform(self, texts: list[str]) -> list[dict[int, float]]:
        rows: list[dict[int, float]] = []
        for text in texts:
            counts: Counter[int] = Counter()
            for term in self._terms(text):
                index = self.vocabulary_.get(term)
                if index is not None:
                    counts[index] += 1

            row: dict[int, float] = {}
            for index, count in counts.items():
                row[index] = (1.0 + math.log(count)) * self.idf_[index]

            norm = math.sqrt(sum(value * value for value in row.values()))
            if norm > 0:
                row = {index: value / norm for index, value in row.items()}
            rows.append(row)
        return rows

    def fit_transform(self, texts: list[str]) -> list[dict[int, float]]:
        return self.fit(texts).transform(texts)

    def to_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "vocabulary": self.vocabulary_,
            "idf": self.idf_,
            "feature_names": self.feature_names_,
            "n_docs": self.n_docs_,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "TfidfVectorizerScratch":
        vectorizer = cls(FeatureConfig(**payload["config"]))
        vectorizer.vocabulary_ = {str(term): int(index) for term, index in payload["vocabulary"].items()}
        vectorizer.idf_ = [float(value) for value in payload["idf"]]
        vectorizer.feature_names_ = [str(value) for value in payload["feature_names"]]
        vectorizer.n_docs_ = int(payload["n_docs"])
        return vectorizer


class ManualSQLFeatureExtractor:
    """Handcrafted SQLi signals, standardized using training data only."""

    BASE_FEATURE_NAMES = (
        "length_chars",
        "length_tokens",
        "digit_count",
        "alpha_count",
        "sql_symbol_count",
        "single_quote_count",
        "double_quote_count",
        "semicolon_count",
        "dashdash_count",
        "block_comment_count",
        "equals_count",
        "parentheses_count",
        "comma_count",
        "percent_count",
        "at_count",
        "hash_count",
        "dot_count",
        "operator_count",
        "keyword_total",
        "symbol_ratio",
        "keyword_density",
        "has_union_select",
        "has_tautology",
        "has_comment",
        "has_stacked_query",
        "starts_with_sql_keyword",
        "has_quote_equals",
        "has_hex_literal",
        "has_encoded_payload",
        "has_xp_cmdshell",
        "has_sleep_or_benchmark",
        "has_schema_probe",
        "has_comment_after_condition",
    )

    def __init__(self):
        self.feature_names_: list[str] = list(self.BASE_FEATURE_NAMES) + [
            f"keyword_{keyword}" for keyword in SQL_KEYWORDS
        ]
        self.means_: list[float] = []
        self.stds_: list[float] = []

    @staticmethod
    def _has_tautology(text: str) -> bool:
        patterns = (
            r"\b(?:or|and)\b\s+['\"]?(\d+)['\"]?\s*=\s*['\"]?\1['\"]?",
            r"\b(?:or|and)\b\s+['\"]?([a-z])['\"]?\s*=\s*['\"]?\1['\"]?",
            r"\b(\d+)\s*=\s*\1\b",
            r"['\"]([a-z])['\"]\s*=\s*['\"]\1['\"]",
        )
        return any(re.search(pattern, text) for pattern in patterns)

    @staticmethod
    def _raw_features(text: str) -> list[float]:
        cleaned = clean_text(text)
        tokens = tokenize(cleaned)
        counts = Counter(tokens)
        keyword_total = sum(counts[keyword] for keyword in SQL_KEYWORDS)
        length_chars = len(cleaned)
        length_tokens = len(tokens)
        sql_symbol_count = sum(1 for char in cleaned if char in "'\"=<>-;*/(),#@+%._")
        operator_count = sum(1 for char in cleaned if char in "=<>+-*/%")
        has_union_select = bool(re.search(r"\bunion\b\s+(?:all\s+)?\bselect\b", cleaned))
        has_comment = "--" in cleaned or "#" in cleaned or "/*" in cleaned
        has_stacked_query = bool(
            re.search(r";\s*(select|drop|insert|update|delete|exec|alter|create)\b", cleaned)
        )
        starts_with_sql_keyword = bool(tokens and tokens[0] in SQL_KEYWORDS)
        has_quote_equals = bool(re.search(r"['\"][^'\"]{0,40}=", cleaned))
        has_hex_literal = bool(re.search(r"\b0x[0-9a-f]+\b", cleaned))
        has_encoded_payload = any(marker in cleaned for marker in ("%27", "%22", "%3d", "%2d", "%2f"))
        has_xp_cmdshell = "xp_cmdshell" in cleaned
        has_sleep_or_benchmark = "sleep" in counts or "benchmark" in counts
        has_schema_probe = "information_schema" in cleaned or "@@version" in cleaned
        has_comment_after_condition = bool(re.search(r"\b(or|and)\b.*(--|#|/\*)", cleaned))

        base = [
            float(length_chars),
            float(length_tokens),
            float(sum(char.isdigit() for char in cleaned)),
            float(sum(char.isalpha() for char in cleaned)),
            float(sql_symbol_count),
            float(cleaned.count("'")),
            float(cleaned.count('"')),
            float(cleaned.count(";")),
            float(cleaned.count("--")),
            float(cleaned.count("/*") + cleaned.count("*/")),
            float(cleaned.count("=")),
            float(cleaned.count("(") + cleaned.count(")")),
            float(cleaned.count(",")),
            float(cleaned.count("%")),
            float(cleaned.count("@")),
            float(cleaned.count("#")),
            float(cleaned.count(".")),
            float(operator_count),
            float(keyword_total),
            sql_symbol_count / max(1.0, float(length_chars)),
            keyword_total / max(1.0, float(length_tokens)),
            float(has_union_select),
            float(ManualSQLFeatureExtractor._has_tautology(cleaned)),
            float(has_comment),
            float(has_stacked_query),
            float(starts_with_sql_keyword),
            float(has_quote_equals),
            float(has_hex_literal),
            float(has_encoded_payload),
            float(has_xp_cmdshell),
            float(has_sleep_or_benchmark),
            float(has_schema_probe),
            float(has_comment_after_condition),
        ]
        base.extend(float(counts[keyword]) for keyword in SQL_KEYWORDS)
        return base

    def fit(self, texts: list[str]) -> "ManualSQLFeatureExtractor":
        rows = [self._raw_features(text) for text in texts]
        width = len(self.feature_names_)
        self.means_ = []
        self.stds_ = []
        for column in range(width):
            values = [row[column] for row in rows]
            mean = sum(values) / max(1, len(values))
            variance = sum((value - mean) ** 2 for value in values) / max(1, len(values))
            std = math.sqrt(variance)
            self.means_.append(mean)
            self.stds_.append(std if std > 1e-12 else 1.0)
        return self

    def transform(self, texts: list[str]) -> list[dict[int, float]]:
        rows: list[dict[int, float]] = []
        for text in texts:
            raw = self._raw_features(text)
            scaled = {
                index: (value - self.means_[index]) / self.stds_[index]
                for index, value in enumerate(raw)
            }
            rows.append(scaled)
        return rows

    def fit_transform(self, texts: list[str]) -> list[dict[int, float]]:
        return self.fit(texts).transform(texts)

    def to_dict(self) -> dict:
        return {
            "feature_names": self.feature_names_,
            "means": self.means_,
            "stds": self.stds_,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ManualSQLFeatureExtractor":
        extractor = cls()
        extractor.feature_names_ = [str(value) for value in payload["feature_names"]]
        extractor.means_ = [float(value) for value in payload["means"]]
        extractor.stds_ = [float(value) for value in payload["stds"]]
        return extractor


class CombinedFeatureExtractor:
    """Combines sparse TF-IDF terms with standardized manual SQL features."""

    def __init__(self, config: FeatureConfig | None = None):
        self.config = config or FeatureConfig()
        self.tfidf = TfidfVectorizerScratch(self.config)
        self.manual = ManualSQLFeatureExtractor()
        self.n_features_: int = 0

    def fit(self, texts: list[str]) -> "CombinedFeatureExtractor":
        self.tfidf.fit(texts)
        self.manual.fit(texts)
        self.n_features_ = len(self.tfidf.feature_names_) + len(self.manual.feature_names_)
        return self

    def transform(self, texts: list[str]) -> list[dict[int, float]]:
        tfidf_rows = self.tfidf.transform(texts)
        manual_rows = self.manual.transform(texts)
        offset = len(self.tfidf.feature_names_)
        combined_rows: list[dict[int, float]] = []
        for tfidf_row, manual_row in zip(tfidf_rows, manual_rows):
            row = dict(tfidf_row)
            for index, value in manual_row.items():
                if abs(value) > 1e-12:
                    row[offset + index] = value
            combined_rows.append(row)
        return combined_rows

    def fit_transform(self, texts: list[str]) -> list[dict[int, float]]:
        return self.fit(texts).transform(texts)

    def feature_names(self) -> list[str]:
        return [f"tfidf:{name}" for name in self.tfidf.feature_names_] + [
            f"manual:{name}" for name in self.manual.feature_names_
        ]

    def to_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "tfidf": self.tfidf.to_dict(),
            "manual": self.manual.to_dict(),
            "n_features": self.n_features_,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "CombinedFeatureExtractor":
        extractor = cls(FeatureConfig(**payload["config"]))
        extractor.tfidf = TfidfVectorizerScratch.from_dict(payload["tfidf"])
        extractor.manual = ManualSQLFeatureExtractor.from_dict(payload["manual"])
        extractor.n_features_ = int(payload["n_features"])
        return extractor
