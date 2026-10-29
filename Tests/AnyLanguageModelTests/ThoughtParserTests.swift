import XCTest
#if Llama
    @testable import AnyLanguageModel

    final class ThoughtParserTests: XCTestCase {
        private func parseAll(_ chunks: [String]) -> String {
            var parser = ThoughtParser()
            for chunk in chunks {
                parser.consume(chunk)
            }
            return parser.thinking
        }

        func testThinkSingleChunk() {
            let thinking = parseAll(["先想：<think>9.9 更大</think>，所以答案是 9.9"])
            XCTAssertEqual(thinking, "9.9 更大")
        }

        func testStartOfThinkPair() {
            let thinking = parseAll(["旧版：<|startofthink|>内部推理</|endofthink|>，结束"])
            XCTAssertEqual(thinking, "内部推理")
        }

        func testBeginOfThoughtPair() {
            let thinking = parseAll(["Llama4：<|begin_of_thought|>推理 A</|end_of_thought|>，结束"])
            XCTAssertEqual(thinking, "推理 A")
        }

        func testReasoningPair() {
            let thinking = parseAll(["<reasoning>步骤1</reasoning>答案"])
            XCTAssertEqual(thinking, "步骤1")
        }

        func testThinkingPair() {
            let thinking = parseAll(["<thinking>思考中</thinking>正文"])
            XCTAssertEqual(thinking, "思考中")
        }

        func testOpenTagAcrossChunks() {
            let thinking = parseAll(["<th", "ink>跨块思考</think>", "正文"])
            XCTAssertEqual(thinking, "跨块思考")
        }

        func testCloseTagAcrossChunks() {
            let thinking = parseAll(["<think>思考", "内容</th", "ink>正文"])
            XCTAssertEqual(thinking, "思考内容")
        }

        func testLongThinkingStreamsWhileGenerating() {
            // 长思考段：超过尾部保留窗口的内容在生成中即流出，末尾靠 finalize 补齐
            var parser = ThoughtParser()
            parser.consume("<think>这是比较长的思考内容超过保留窗口")
            XCTAssertTrue(parser.thinking.contains("这是比较长的思考内"), "thinking=[\(parser.thinking)]")
            parser.consume("继续思考")
            parser.finalize()
            XCTAssertEqual(parser.thinking, "这是比较长的思考内容超过保留窗口继续思考")
        }

        func testUnclosedFlushedOnFinalize() {
            // 生成结束：未闭合思考段完整 flush
            var parser = ThoughtParser()
            parser.consume("<think>完整")
            parser.consume("思考内容")
            parser.finalize()
            XCTAssertEqual(parser.thinking, "完整思考内容")
        }

        func testPlainTextNoThinking() {
            let thinking = parseAll(["没有思考标签的", "普通回复"])
            XCTAssertEqual(thinking, "")
        }

        func testBodyTextIgnored() {
            let parser = ThoughtParser()
            var p = parser
            p.consume("这是正文")
            XCTAssertEqual(p.thinking, "")
        }

        func testMultiplePairsSequential() {
            let thinking = parseAll([
                "<think>第一段</think>",
                "中间正文",
                "<|startofthink|>第二段</|endofthink|>",
                "结尾",
            ])
            XCTAssertEqual(thinking, "第一段第二段")
        }
    }
#endif
