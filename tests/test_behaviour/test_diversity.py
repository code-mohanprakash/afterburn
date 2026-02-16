"""Tests for EAD diversity analysis and SBERT semantic diversity."""


from afterburn.behaviour.diversity import (
    DiversityAnalysis,
    _compute_ead,
    _estimate_vocab_sizes,
    _extract_ngrams,
    analyze_diversity,
    compute_semantic_diversity,
)


class TestExtractNgrams:
    def test_unigrams(self):
        tokens = ["hello", "world", "foo"]
        result = _extract_ngrams(tokens, 1)
        assert result == [("hello",), ("world",), ("foo",)]

    def test_bigrams(self):
        tokens = ["a", "b", "c", "d"]
        result = _extract_ngrams(tokens, 2)
        assert result == [("a", "b"), ("b", "c"), ("c", "d")]

    def test_empty(self):
        assert _extract_ngrams([], 1) == []

    def test_too_short(self):
        assert _extract_ngrams(["a"], 2) == []


class TestComputeEAD:
    """Test EAD = D / (V * (1 - ((V-1)/V)^T)) from Liu et al. ACL 2022."""

    def test_empty_outputs(self):
        assert _compute_ead([], 1, vocab_size=100) == 0.0

    def test_single_word_outputs(self):
        outputs = ["hello", "hello", "hello"]
        # V=1 since only one unique unigram, but we pass a reasonable V
        ead = _compute_ead(outputs, 1, vocab_size=10)
        assert ead >= 0.0

    def test_diverse_outputs_higher_than_repetitive(self):
        diverse = [
            "the cat sat on the mat",
            "a dog ran through the park",
            "birds flew over the mountain",
            "fish swam in the deep ocean",
            "horses galloped across the plains",
        ]
        repetitive = [
            "the answer is 42",
            "the answer is 42",
            "the answer is 42",
            "the answer is 42",
            "the answer is 42",
        ]
        # Use the same vocab size for fair comparison (union of all unigrams)
        all_tokens: set[str] = set()
        for text in diverse + repetitive:
            all_tokens.update(text.lower().split())
        V = len(all_tokens)
        diverse_ead = _compute_ead(diverse, 1, vocab_size=V)
        repetitive_ead = _compute_ead(repetitive, 1, vocab_size=V)
        assert diverse_ead > repetitive_ead

    def test_bigram_diversity(self):
        diverse = [
            "machine learning is fascinating",
            "natural language processing works well",
            "computer vision detects objects",
            "reinforcement learning trains agents",
        ]
        ead = _compute_ead(diverse, 2, vocab_size=50)
        assert ead >= 0.0

    def test_perfect_uniformity(self):
        """When all n-grams are unique and V equals D, EAD should be close to 1."""
        # 5 outputs each with unique words
        outputs = ["alpha", "beta", "gamma", "delta", "epsilon"]
        # V = 5 unigrams, D = 5 distinct, T = 5 total
        ead = _compute_ead(outputs, 1, vocab_size=5)
        # D=5, E[D] = 5 * (1 - (4/5)^5) = 5 * (1 - 0.32768) = 5 * 0.67232 = 3.3616
        # EAD = 5 / 3.3616 ≈ 1.487 (more diverse than uniform random!)
        assert ead > 1.0

    def test_formula_matches_manual_calculation(self):
        """Verify EAD formula: D / (V * (1 - ((V-1)/V)^T))."""
        outputs = ["a b c", "a b d", "a e f"]
        # tokens: a b c a b d a e f → 9 total unigrams
        # distinct: {a, b, c, d, e, f} = 6
        V = 10
        ead = _compute_ead(outputs, 1, vocab_size=V)
        # Manual: D=6, T=9, V=10
        # E[D] = 10 * (1 - (9/10)^9) = 10 * (1 - 0.38742) = 10 * 0.61258 = 6.1258
        # EAD = 6 / 6.1258 ≈ 0.9795
        expected = 6 / (10 * (1 - (9 / 10) ** 9))
        assert abs(ead - expected) < 1e-6


class TestEstimateVocabSizes:
    def test_combines_both_output_sets(self):
        base = ["hello world"]
        trained = ["foo bar"]
        vs = _estimate_vocab_sizes(base, trained, max_n=1)
        # Union: {hello, world, foo, bar} = 4
        assert vs[1] == 4

    def test_fixed_vocab_size(self):
        base = ["hello"]
        trained = ["world"]
        vs = _estimate_vocab_sizes(base, trained, max_n=2, fixed_vocab_size=50000)
        assert vs[1] == 50000
        assert vs[2] == 50000

    def test_empty_outputs(self):
        vs = _estimate_vocab_sizes([], [], max_n=3)
        for n in range(1, 4):
            assert vs[n] >= 1  # min is 1


class TestAnalyzeDiversity:
    def test_basic_analysis(self):
        base = ["simple answer one", "simple answer two", "simple answer three"]
        trained = ["detailed response one", "detailed response two", "detailed response three"]
        result = analyze_diversity(base, trained)
        assert isinstance(result, DiversityAnalysis)
        assert 1 in result.base_ead
        assert 1 in result.trained_ead

    def test_ead_keys_1_through_5(self):
        outputs = ["the quick brown fox"] * 5
        result = analyze_diversity(outputs, outputs)
        for n in range(1, 6):
            assert n in result.base_ead
            assert n in result.trained_ead

    def test_diversity_change_sign(self):
        """More diverse trained outputs should have positive diversity_change."""
        base = ["the answer is 42"] * 10
        trained = [
            "the answer is 42",
            "let me think step by step about this problem",
            "using the quadratic formula we get x equals",
            "first we need to consider the base case",
            "the result can be computed as follows",
            "by mathematical induction we prove that",
            "applying the chain rule we find the derivative",
            "the integral evaluates to pi over two",
            "consider the contrapositive of the statement",
            "by the fundamental theorem of calculus",
        ]
        result = analyze_diversity(base, trained)
        assert result.trained_diversity_score >= result.base_diversity_score

    def test_identical_outputs_no_change(self):
        outputs = ["hello world this is a test"] * 5
        result = analyze_diversity(outputs, outputs)
        assert abs(result.diversity_change) < 0.01

    def test_per_category(self):
        base = ["answer one", "answer two", "code one", "code two"]
        trained = ["long detailed answer one", "long detailed answer two",
                   "def foo(): pass", "class Bar: pass"]
        cats = ["math", "math", "code", "code"]
        result = analyze_diversity(base, trained, cats, cats)
        assert "math" in result.per_category or "code" in result.per_category

    def test_empty_inputs(self):
        result = analyze_diversity([], [])
        assert result.base_diversity_score == 0.0
        assert result.trained_diversity_score == 0.0

    def test_single_output(self):
        result = analyze_diversity(["hello world"], ["goodbye world"])
        assert isinstance(result, DiversityAnalysis)

    def test_diversity_drop_detected(self):
        """RLHF-style mode collapse: diverse base → repetitive trained."""
        base = [
            "There are several approaches to this problem",
            "One way to solve this is through dynamic programming",
            "We can use a greedy algorithm here",
            "The brute force solution iterates over all pairs",
            "A recursive approach with memoization works well",
            "Binary search can find the answer efficiently",
            "Using a hash map reduces the time complexity",
            "The sliding window technique applies here",
        ]
        trained = [
            "Let me solve this step by step. First, we identify the pattern.",
            "Let me solve this step by step. First, we set up the equation.",
            "Let me solve this step by step. First, we define variables.",
            "Let me solve this step by step. First, we consider the input.",
            "Let me solve this step by step. First, we analyze the structure.",
            "Let me solve this step by step. First, we break down the problem.",
            "Let me solve this step by step. First, we examine the constraints.",
            "Let me solve this step by step. First, we outline our approach.",
        ]
        result = analyze_diversity(base, trained)
        # Trained starts with the same phrase every time — less diverse
        assert result.base_diversity_score > result.trained_diversity_score
        assert result.diversity_change < 0

    def test_semantic_diversity_fields_present(self):
        """Semantic diversity fields should exist (None if no sentence-transformers)."""
        result = analyze_diversity(["hello world"], ["goodbye world"])
        assert hasattr(result, "base_semantic_diversity")
        assert hasattr(result, "trained_semantic_diversity")
        assert hasattr(result, "semantic_diversity_change")


class TestSemanticDiversity:
    def test_returns_none_or_float(self):
        """Should return None (no package) or float (package installed)."""
        result = compute_semantic_diversity(["hello", "world"])
        assert result is None or isinstance(result, float)

    def test_single_output_returns_zero(self):
        result = compute_semantic_diversity(["hello world"])
        # Not enough outputs for pairwise comparison
        assert result == 0.0

    def test_empty_returns_none(self):
        """Empty list has < 2 outputs."""
        result = compute_semantic_diversity([])
        assert result == 0.0
