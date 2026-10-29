import XCTest
#if Llama
    @testable import AnyLanguageModel

    final class ThoughtSpanDetectionTests: XCTestCase {
        func testQwen3NewTemplate() {
            let template = """
            {% if not is_tool_call %}{{'<think>\n'}}{% endif %}<|im_start|>assistant
            """
            let pairs = ThoughtParser.thoughtSpanPairs(fromTemplate: template)
            XCTAssertEqual(pairs?.count, 1)
            XCTAssertEqual(pairs?.first?.open, "<think>")
            XCTAssertEqual(pairs?.first?.close, "</think>")
        }

        func testQwen2OldStyleTemplate() {
            let template = """
            <|im_start|>system ... <|startofthink|>{{ {content} }}<|endofthink|>
            """
            let pairs = ThoughtParser.thoughtSpanPairs(fromTemplate: template)
            XCTAssertEqual(pairs?.count, 1)
            XCTAssertEqual(pairs?.first?.open, "<|startofthink|>")
            XCTAssertEqual(pairs?.first?.close, "</|endofthink|>")
        }

        func testLlama4Template() {
            let template = """
            <|begin_of_thought|>{{{ content }}}<|end_of_thought|>
            """
            let pairs = ThoughtParser.thoughtSpanPairs(fromTemplate: template)
            XCTAssertEqual(pairs?.count, 1)
            XCTAssertEqual(pairs?.first?.open, "<|begin_of_thought|>")
            XCTAssertEqual(pairs?.first?.close, "</|end_of_thought|>")
        }

        func testGenericPairDiscovery() {
            // 模板含 <reasoning>...</reasoning>，未在已知集合中显式出现于模板外
            let template = "{% if reasoning %}<reasoning>{{ content }}</reasoning>{% endif %}"
            let pairs = ThoughtParser.thoughtSpanPairs(fromTemplate: template)
            XCTAssertEqual(pairs?.count, 1)
            XCTAssertEqual(pairs?.first?.open, "<reasoning>")
            XCTAssertEqual(pairs?.first?.close, "</reasoning>")
        }

        func testTemplateWithoutThinking() {
            let template = "<|im_start|>user {{ content }}<|im_end|>"
            let pairs = ThoughtParser.thoughtSpanPairs(fromTemplate: template)
            XCTAssertNil(pairs)
        }

        func testMultipleKnownPairs() {
            let template = """
            <thinking>{{ content }}</thinking><think>{% if x %}</think>
            """
            let pairs = ThoughtParser.thoughtSpanPairs(fromTemplate: template)
            XCTAssertEqual(pairs?.count, 2)
        }

        func testDefaultPairsFallback() {
            // 默认兜底全集应覆盖所有已知思考标签
            XCTAssertEqual(ThoughtParser.defaultPairs.count, 5)
        }
    }
#endif
