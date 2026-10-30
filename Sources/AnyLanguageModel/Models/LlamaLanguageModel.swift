import Foundation
#if os(macOS)
    import Darwin
    import Darwin.Mach
#endif
#if Llama
    import JSONSchema
    import LlamaSwift

    /// Global storage for the current log level threshold.
    /// This is needed because the C callback can't capture Swift context.
    /// Access is synchronized by llama.cpp's internal logging mechanism.
    nonisolated(unsafe) private var currentLogLevel: LlamaLanguageModel.LogLevel = .warn

    /// Custom log callback that filters messages based on the current log level.
    private func llamaLogCallback(
        level: ggml_log_level,
        text: UnsafePointer<CChar>?,
        userData: UnsafeMutableRawPointer?
    ) {
        guard level.rawValue >= currentLogLevel.ggmlLevel.rawValue else { return }
        if let text = text {
            fputs(String(cString: text), stderr)
        }
    }

    /// Incrementally parses thinking spans (all common tag pairs) out of raw
    /// generated text, streaming the thinking content separately from the
    /// visible body. Handles tags that span multiple tokens/chunks by
    /// buffering only the trailing tag-prefix.
    struct ThoughtParser {
        /// 兜底标签对全集（模型模板检测失败时使用）
        static let defaultPairs: [(open: String, close: String)] = [
            ("<|startofthink|>", "</|endofthink|>"),
            ("<|begin_of_thought|>", "</|end_of_thought|>"),
            ("<reasoning>", "</reasoning>"),
            ("<thinking>", "</thinking>"),
            ("<think>", "</think>"),
        ]

        private let pairs: [(open: String, close: String)]
        private let maxOpenTagLength: Int

        private(set) var thinking = ""
        private var buffer = ""
        private var activeClose: String?

        /// - Parameter pairs: 思考标签对；缺省使用兜底全集。
        init(pairs: [(open: String, close: String)] = ThoughtParser.defaultPairs) {
            self.pairs = pairs
            self.maxOpenTagLength = pairs.map { $0.open.count }.max() ?? 8
        }

        mutating func consume(_ raw: String) {
            buffer += raw
            while true {
                if let close = activeClose {
                    if let found = buffer.range(of: close) {
                        thinking += buffer[..<found.lowerBound]
                        buffer = String(buffer[found.upperBound...])
                        activeClose = nil
                    } else {
                        let keep = close.count - 1
                        if buffer.count > keep {
                            let cut = buffer.index(buffer.endIndex, offsetBy: -keep)
                            thinking += buffer[..<cut]
                            buffer = String(buffer[cut...])
                        } else {
                            break
                        }
                    }
                } else {
                    // 非思考：取最早出现的 open 标签
                    var best: (range: Range<String.Index>, close: String)?
                    for pair in self.pairs {
                        if let r = buffer.range(of: pair.open),
                            best == nil || r.lowerBound < best!.range.lowerBound {
                            best = (r, pair.close)
                        }
                    }
                    if let best {
                        buffer = String(buffer[best.range.upperBound...])
                        activeClose = best.close
                    } else {
                        // 保留最长 open 标签前缀，等待下一 chunk 确认
                        let keep = self.maxOpenTagLength - 1
                        if buffer.count > keep {
                            let cut = buffer.index(buffer.endIndex, offsetBy: -keep)
                            buffer = String(buffer[cut...])
                        } else {
                            break
                        }
                    }
                }
            }
        }

        /// 生成结束：flush 剩余 buffer（未闭合的思考段也完整输出）
        mutating func finalize() {
            thinking += buffer
            buffer = ""
            activeClose = nil
        }
    }

extension ThoughtParser {
    /// 从 llama 模型元数据（tokenizer.chat_template）动态检测思考标签对，
    /// 无需针对每个模型写死标签。查不到模板或无思考标签时返回 nil（用兜底全集）。
    static func detectThoughtSpans(model: OpaquePointer?) -> [(open: String, close: String)]? {
        guard let model else { return nil }
        var buffer = [CChar](repeating: 0, count: 65536)
        let count = llama_model_meta_val_str(
            model, "tokenizer.chat_template", &buffer, buffer.count
        )
        guard count > 0 else { return nil }
        let template = String(cString: buffer)
        return thoughtSpanPairs(fromTemplate: template)
    }

    /// 从 chat template 文本提取思考标签对：
    /// 1) 已知标签集合匹配（模板里有哪个用哪个，全无则跳过）；
    /// 2) 通用成对发现：<X>…</X> 且 X 含 think/thought/reasoning 关键词。
    static func thoughtSpanPairs(fromTemplate template: String) -> [(open: String, close: String)]? {
        let known: [(open: String, close: String)] = [
            ("<|startofthink|>", "</|endofthink|>"),
            ("<|begin_of_thought|>", "</|end_of_thought|>"),
            ("<reasoning>", "</reasoning>"),
            ("<thinking>", "</thinking>"),
            ("<think>", "</think>"),
        ]
        let matched = known.filter { template.contains($0.open) }
        if !matched.isEmpty { return matched }

        // 通用成对发现：<X> 与 </X> 同时出现在模板，且 X 含思考关键词
        let pattern = #"<(/?)\|?([A-Za-z_]*?(?:think|thought|reasoning)[A-Za-z_]*?)\|?>"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return nil }
        let ns = template as NSString
        var openTags: Set<String> = []
        var closeTags: Set<String> = []
        for match in regex.matches(in: template, range: NSRange(location: 0, length: ns.length)) {
            if let range = Range(match.range, in: template) {
                let tag = String(template[range])
                if tag.hasPrefix("</") {
                    closeTags.insert(tag)
                } else {
                    openTags.insert(tag)
                }
            }
        }
        var pairs: [(open: String, close: String)] = []
        for open in openTags {
            let close = "</" + open.dropFirst()
            if closeTags.contains(close) {
                pairs.append((open, close))
            }
        }
        return pairs.isEmpty ? nil : pairs
    }
}

    /// A language model that runs llama.cpp models locally.
    ///
    /// Use this model to generate text using GGUF models running directly with llama.cpp.
    ///
    /// ```swift
    /// let model = LlamaLanguageModel(modelPath: "/path/to/model.gguf")
    /// ```
    public final class LlamaLanguageModel: LanguageModel, @unchecked Sendable {
        /// The reason the model is unavailable.
        /// This model is always available.
        public typealias UnavailableReason = Never

        /// The verbosity level for llama.cpp logging.
        public enum LogLevel: Int, Hashable, Comparable, Sendable, CaseIterable {
            /// No logging output.
            case none = 0
            /// Debug messages and above (most verbose).
            case debug = 1
            /// Info messages and above.
            case info = 2
            /// Warning messages and above (default).
            case warn = 3
            /// Only error messages.
            case error = 4

            /// Maps to the corresponding ggml log level.
            var ggmlLevel: ggml_log_level {
                switch self {
                case .none: return GGML_LOG_LEVEL_NONE
                case .debug: return GGML_LOG_LEVEL_DEBUG
                case .info: return GGML_LOG_LEVEL_INFO
                case .warn: return GGML_LOG_LEVEL_WARN
                case .error: return GGML_LOG_LEVEL_ERROR
                }
            }

            public static func < (lhs: LogLevel, rhs: LogLevel) -> Bool {
                lhs.rawValue < rhs.rawValue
            }
        }

        /// Custom generation options specific to llama.cpp.
        ///
        /// Reached through `GenerationOptions[custom: LlamaLanguageModel.self]`,
        /// an AnyLanguageModel extension.
        ///
        /// Use this type to pass llama.cpp-specific sampling parameters that are
        /// not part of the standard ``GenerationOptions``.
        ///
        /// ```swift
        /// var options = GenerationOptions(temperature: 0.8)
        /// options[custom: LlamaLanguageModel.self] = .init(
        ///     repeatPenalty: 1.2,
        ///     repeatLastN: 128,
        ///     frequencyPenalty: 0.1,
        ///     presencePenalty: 0.1,
        ///     mirostat: .v2(tau: 5.0, eta: 0.1)
        /// )
        /// ```
        public struct CustomGenerationOptions: AnyLanguageModel.CustomGenerationOptions, Codable {
            /// Context size to allocate for the model.
            public var contextSize: UInt32?

            /// Batch size to use when evaluating tokens.
            public var batchSize: UInt32?

            /// Number of threads to use for computation.
            public var threads: Int32?

            /// Random seed for deterministic sampling.
            public var seed: UInt32?

            /// Sampling temperature.
            public var temperature: Float?

            /// Top-K sampling parameter.
            public var topK: Int32?

            /// Top-P (nucleus) sampling parameter.
            public var topP: Float?

            /// The penalty applied to repeated tokens.
            ///
            /// Values greater than 1.0 discourage repetition, while values less than 1.0
            /// encourage it. A value of 1.0 applies no penalty.
            public var repeatPenalty: Float?

            /// The number of recent tokens to consider for the repeat penalty.
            ///
            /// Only the last `repeatLastN` tokens will be checked for repetition.
            /// Set to 0 to disable repeat penalty, or -1 to consider all tokens.
            public var repeatLastN: Int32?

            /// The frequency penalty applied during sampling.
            ///
            /// Positive values penalize tokens based on their frequency in the text so far,
            /// decreasing the likelihood of repeating the same content.
            public var frequencyPenalty: Float?

            /// The presence penalty applied during sampling.
            ///
            /// Positive values penalize tokens that have appeared at all in the text so far,
            /// encouraging the model to generate novel content.
            public var presencePenalty: Float?

            /// Mirostat sampling configuration for adaptive perplexity control.
            public enum MirostatMode: Hashable, Codable, Sendable {
                /// Mirostat v1 with target entropy (tau) and learning rate (eta).
                case v1(tau: Float, eta: Float)

                /// Mirostat v2 with target entropy (tau) and learning rate (eta).
                case v2(tau: Float, eta: Float)
            }

            /// Mirostat sampling mode for adaptive perplexity control.
            public var mirostat: MirostatMode?

            /// Text appended after the assistant header of the rendered prompt.
            ///
            /// The model continues generating from this text.
            /// Use it to steer the start of the response,
            /// for example by prefilling an empty `<think></think>` block
            /// to suppress a model's default reasoning output
            /// when its chat template offers no switch for it.
            public var assistantPrefill: String?

            /// Creates custom generation options for llama.cpp.
            public init(
                contextSize: UInt32? = nil,
                batchSize: UInt32? = nil,
                threads: Int32? = nil,
                seed: UInt32? = nil,
                temperature: Float? = nil,
                topK: Int32? = nil,
                topP: Float? = nil,
                repeatPenalty: Float? = nil,
                repeatLastN: Int32? = nil,
                frequencyPenalty: Float? = nil,
                presencePenalty: Float? = nil,
                mirostat: MirostatMode? = nil,
                assistantPrefill: String? = nil
            ) {
                self.contextSize = contextSize
                self.batchSize = batchSize
                self.threads = threads
                self.seed = seed
                self.temperature = temperature
                self.topK = topK
                self.topP = topP
                self.repeatPenalty = repeatPenalty
                self.repeatLastN = repeatLastN
                self.frequencyPenalty = frequencyPenalty
                self.presencePenalty = presencePenalty
                self.mirostat = mirostat
                self.assistantPrefill = assistantPrefill
            }

            /// Default llama.cpp options used when none are provided at runtime.
            ///
            /// The `seed` is `nil` by default, meaning a random seed will be generated
            /// for each generation request.
            public static var `default`: Self {
                .init(
                    contextSize: 4096,
                    batchSize: 512,
                    threads: LlamaLanguageModel.recommendedThreadCount(),
                    seed: nil,
                    temperature: 0.8,
                    topK: 40,
                    topP: 0.95,
                    repeatPenalty: 1.1,
                    repeatLastN: 64,
                    frequencyPenalty: 0.0,
                    presencePenalty: 0.0,
                    mirostat: nil
                )
            }

        }

        /// The path to the GGUF model file.
        public let modelPath: String

        /// The number of model layers to offload to the GPU.
        ///
        /// A negative value offloads all layers, and `0` runs entirely on the CPU.
        public let gpuLayers: Int32

        /// The path to the multimodal projector GGUF file, when the model has one.
        ///
        /// Prompts may include image segments only when a projector is loaded.
        public let mmprojPath: String?

        /// The default GPU layer count for the current platform.
        ///
        /// All layers are offloaded by default: the prebuilt llama.cpp binaries
        /// ship with Metal enabled and the shader library embedded. The simulator
        /// defaults to CPU-only execution, which remains the reliable
        /// configuration there.
        public static var defaultGPULayerCount: Int32 {
            #if targetEnvironment(simulator)
                return 0
            #else
                return -1
            #endif
        }

        /// The default options that each request's options override.
        private let defaultOptions: ResolvedGenerationOptions

        /// The minimum log level for llama.cpp output.
        ///
        /// This is a global setting that affects all `LlamaLanguageModel` instances
        /// since llama.cpp uses a single global log callback.
        public nonisolated(unsafe) static var logLevel: LogLevel = .warn {
            didSet {
                currentLogLevel = logLevel
                llama_log_set(llamaLogCallback, nil)
            }
        }

        /// Resolved, non-optional defaults for llama.cpp runtime parameters.
        internal struct ResolvedGenerationOptions: Sendable {
            var contextSize: UInt32
            var batchSize: UInt32
            var threads: Int32
            var seed: UInt32
            var temperature: Float
            var topK: Int32
            var topP: Float
            var repeatPenalty: Float
            var repeatLastN: Int32
            var frequencyPenalty: Float
            var presencePenalty: Float
            var mirostat: CustomGenerationOptions.MirostatMode?
            var assistantPrefill: String?
            var sampling: GenerationOptions.SamplingMode?
            var maximumResponseTokens: Int?

            init(
                contextSize: UInt32 = 4096,
                batchSize: UInt32 = 512,
                threads: Int32 = LlamaLanguageModel.recommendedThreadCount(),
                seed: UInt32 = UInt32.random(in: 0 ... UInt32.max),
                temperature: Float = 0.8,
                topK: Int32 = 40,
                topP: Float = 0.95,
                repeatPenalty: Float = 1.1,
                repeatLastN: Int32 = 64,
                frequencyPenalty: Float = 0.0,
                presencePenalty: Float = 0.0,
                mirostat: CustomGenerationOptions.MirostatMode? = nil,
                assistantPrefill: String? = nil,
                sampling: GenerationOptions.SamplingMode? = nil,
                maximumResponseTokens: Int? = nil
            ) {
                self.contextSize = contextSize
                self.batchSize = batchSize
                self.threads = threads
                self.seed = seed
                self.temperature = temperature
                self.topK = topK
                self.topP = topP
                self.repeatPenalty = repeatPenalty
                self.repeatLastN = repeatLastN
                self.frequencyPenalty = frequencyPenalty
                self.presencePenalty = presencePenalty
                self.mirostat = mirostat
                self.assistantPrefill = assistantPrefill
                self.sampling = sampling
                self.maximumResponseTokens = maximumResponseTokens
            }

            init(
                from options: CustomGenerationOptions?,
                sampling: GenerationOptions.SamplingMode? = nil,
                maximumResponseTokens: Int? = nil
            ) {
                self.init(
                    base: ResolvedGenerationOptions(),
                    overrides: options,
                    sampling: sampling,
                    maximumResponseTokens: maximumResponseTokens
                )
            }

            init(
                base: ResolvedGenerationOptions = .init(),
                overrides options: CustomGenerationOptions?,
                sampling: GenerationOptions.SamplingMode? = nil,
                maximumResponseTokens: Int? = nil
            ) {
                guard let options else {
                    self = ResolvedGenerationOptions(
                        contextSize: base.contextSize,
                        batchSize: base.batchSize,
                        threads: base.threads,
                        seed: base.seed,
                        temperature: base.temperature,
                        topK: base.topK,
                        topP: base.topP,
                        repeatPenalty: base.repeatPenalty,
                        repeatLastN: base.repeatLastN,
                        frequencyPenalty: base.frequencyPenalty,
                        presencePenalty: base.presencePenalty,
                        mirostat: base.mirostat,
                        assistantPrefill: base.assistantPrefill,
                        sampling: sampling ?? base.sampling,
                        maximumResponseTokens: maximumResponseTokens ?? base.maximumResponseTokens
                    )
                    return
                }

                self.contextSize = options.contextSize ?? base.contextSize
                self.batchSize = options.batchSize ?? base.batchSize
                self.threads = options.threads ?? base.threads
                self.seed = options.seed ?? base.seed
                self.temperature = options.temperature ?? base.temperature
                self.topK = options.topK ?? base.topK
                self.topP = options.topP ?? base.topP
                self.repeatPenalty = options.repeatPenalty ?? base.repeatPenalty
                self.repeatLastN = options.repeatLastN ?? base.repeatLastN
                self.frequencyPenalty = options.frequencyPenalty ?? base.frequencyPenalty
                self.presencePenalty = options.presencePenalty ?? base.presencePenalty
                self.mirostat = options.mirostat ?? base.mirostat
                self.assistantPrefill = options.assistantPrefill ?? base.assistantPrefill
                self.sampling = sampling ?? base.sampling
                self.maximumResponseTokens = maximumResponseTokens ?? base.maximumResponseTokens
            }
        }

        /// The loaded model instance
        private var model: OpaquePointer?
        /// 模型原生训练上下文长度（llama_n_ctx_train；GGUF 自带 context_length）
        private var nativeContextSize: UInt32?
        /// 从模型 chat template 动态检测出的思考标签对（nil = 用兜底全集）
        private var detectedThoughtPairs: [(open: String, close: String)]?
        /// The model's vocabulary
        private var vocab: OpaquePointer?

        /// Vocabulary-derived tokens shared across generations of this loaded model.
        private var tokenCache: StructuredGenerationTokenCache?

        /// The multimodal projector context, when a projector file was provided
        private var mtmdContext: OpaquePointer?
        /// `mtmd_helper_eval_chunks` is not thread-safe and every multimodal generation
        /// shares the one projector context, so bitmap init through eval runs under this lock.
        private let projectorLock = NSLock()

        private func projectorLocked<T>(_ body: () throws -> T) rethrows -> T {
            projectorLock.lock()
            defer { projectorLock.unlock() }
            return try body()
        }

        /// Serializes the loaded-state check, loading, and publication of the model,
        /// vocabulary, and projector so concurrent first requests cannot reload them.
        private let modelLoadLock = NSLock()

        /// Whether the model is currently loaded. Protected by `modelLoadLock`.
        private var isModelLoaded: Bool = false

        /// 加载时的内存估算基线（模型权重 + 缓冲），运行时内存护栏据此计算阈值。
        private var memoryLoadEstimateBytes: UInt64 = 0

        /// 推理串行门：同一模型实例同一时刻只允许一个生成在跑。
        /// 并发 llama_decode 共享同一 context/batch/tokenCache，竞争会导致崩溃或卡死，
        /// 所有生成入口（文本/JSON）都经过这里串行化，后续请求排队而非并发。
        private let inferenceGate = DispatchSemaphore(value: 1)

        /// 最近一次生成的 token/s（性能报告读取；inferenceGate 内串行写、statsLock 保护）
        private var lastTokensPerSecond: Double = 0
        private let statsLock = NSLock()

        /// 结构化 JSON 生成的串行门（async 安全）：JSON 生成与文本生成共享
        /// 同一 model 实例与 tokenCache，必须串行执行。async 上下文不能同步
        /// wait DispatchSemaphore，故用 actor 信号量实现。
        private actor InferenceGate {
            private var waiters: [CheckedContinuation<Void, Never>] = []
            private var locked = false

            func acquire() async {
                if !locked {
                    locked = true
                    return
                }
                await withCheckedContinuation { waiters.append($0) }
            }

            func release() {
                if waiters.isEmpty {
                    locked = false
                } else {
                    waiters.removeFirst().resume()
                }
            }
        }

        private let structuredGate = InferenceGate()

        /// A context kept alive for one session so exchanges reuse its state.
        private struct CachedSessionContext {
            let sessionID: ObjectIdentifier
            let context: OpaquePointer
            var tokens: [llama_token]
            let contextSize: UInt32
            let batchSize: UInt32
            /// Whether a generation is currently decoding on the context.
            var isCheckedOut: Bool
            /// Whether the context should be freed once the current generation releases it.
            var discardWhenReleased: Bool
        }

        /// Guards `cachedSessionContext`. A generation checks the cached context out
        /// for its whole run, so a concurrent generation for another session never
        /// frees a context that is still decoding: it runs on a transient context
        /// instead and leaves the cache untouched.
        private let sessionContextLock = NSLock()
        private var cachedSessionContext: CachedSessionContext?

        /// The number of prompt tokens reused from the cached context by the most
        /// recent chat generation.
        internal private(set) var lastReusedTokenCount: Int = 0

        /// The number of prompt tokens decoded by the most recent chat generation.
        internal private(set) var lastPrefillTokenCount: Int = 0

        /// Frees the cached per-session context and the state it holds.
        ///
        /// The cached context, including its KV state, otherwise lives as long as the
        /// model. The next chat generation prefills its full prompt again. Call this
        /// under memory pressure or when a session is discarded. If a generation is
        /// running on the cached context, it is freed as soon as that generation ends.
        public func clearCachedContext() {
            discardCachedSessionContext()
        }

        private func discardCachedSessionContext() {
            sessionContextLock.lock()
            defer { sessionContextLock.unlock() }
            guard var cached = cachedSessionContext else { return }
            if cached.isCheckedOut {
                cached.discardWhenReleased = true
                cachedSessionContext = cached
                return
            }
            llama_free(cached.context)
            cachedSessionContext = nil
        }

        private func recordCachedTokens(_ tokens: [llama_token], context: OpaquePointer) {
            sessionContextLock.lock()
            defer { sessionContextLock.unlock() }
            guard var cached = cachedSessionContext, cached.context == context else { return }
            cached.tokens = tokens
            cachedSessionContext = cached
        }

        private func isCachedSessionContext(_ context: OpaquePointer) -> Bool {
            sessionContextLock.lock()
            defer { sessionContextLock.unlock() }
            return cachedSessionContext?.context == context
        }

        /// Returns a context obtained from `acquireSessionContext`. The cached
        /// context is checked back in (or freed, if a discard was requested while
        /// it was busy); a transient context is freed.
        private func releaseSessionContext(_ context: OpaquePointer) {
            sessionContextLock.lock()
            defer { sessionContextLock.unlock() }
            if var cached = cachedSessionContext, cached.context == context {
                if cached.discardWhenReleased {
                    llama_free(cached.context)
                    cachedSessionContext = nil
                } else {
                    cached.isCheckedOut = false
                    cachedSessionContext = cached
                }
                return
            }
            llama_free(context)
        }

        /// Returns a context for the session along with the index of the first
        /// prompt token that still needs to be decoded. Pair every call with
        /// `releaseSessionContext(_:)` once generation ends.
        ///
        /// A cached context whose recorded tokens share a prefix with the prompt
        /// keeps that prefix: matching state past the divergence point is removed
        /// with `llama_memory_seq_rm`, and backends whose state cannot be rewound
        /// (recurrent models) fall back to clearing the memory and decoding the
        /// full prompt. The final prompt token is always re-decoded so sampling
        /// has fresh logits.
        ///
        /// While another generation holds the cached context, the caller gets a
        /// transient context that decodes the full prompt and is not cached.
        private func acquireSessionContext(
            for session: LanguageModelSession,
            promptTokens: [llama_token],
            options: ResolvedGenerationOptions
        ) throws -> (context: OpaquePointer, startIndex: Int) {
            let sessionID = ObjectIdentifier(session)
            sessionContextLock.lock()
            defer { sessionContextLock.unlock() }

            if var cached = cachedSessionContext,
                !cached.isCheckedOut,
                cached.sessionID == sessionID,
                cached.contextSize == options.contextSize,
                cached.batchSize == options.batchSize
            {
                var common = 0
                while common < cached.tokens.count, common < promptTokens.count,
                    cached.tokens[common] == promptTokens[common]
                {
                    common += 1
                }
                if common == promptTokens.count {
                    common = max(0, promptTokens.count - 1)
                }
                if common < cached.tokens.count {
                    let memory = llama_get_memory(cached.context)
                    if !llama_memory_seq_rm(memory, 0, llama_pos(common), -1) {
                        llama_memory_clear(memory, true)
                        common = 0
                    }
                }
                cached.tokens = Array(promptTokens.prefix(common))
                cached.isCheckedOut = true
                cachedSessionContext = cached
                return (cached.context, common)
            }

            if let cached = cachedSessionContext, cached.isCheckedOut {
                return (try makeFreshContext(options: options), 0)
            }

            if let cached = cachedSessionContext {
                llama_free(cached.context)
                cachedSessionContext = nil
            }
            let contextParams = createContextParams(from: options)
            guard let context = llama_init_from_model(model!, contextParams) else {
                throw LlamaLanguageModelError.contextInitializationFailed
            }
            guard llama_get_memory(context) != nil else {
                llama_free(context)
                throw LlamaLanguageModelError.encoderOnlyModel
            }
            cachedSessionContext = CachedSessionContext(
                sessionID: sessionID,
                context: context,
                tokens: [],
                contextSize: options.contextSize,
                batchSize: options.batchSize,
                isCheckedOut: true,
                discardWhenReleased: false
            )
            return (context, 0)
        }

        /// Creates a single-use context for generations that do not reuse state.
        private func makeFreshContext(options: ResolvedGenerationOptions) throws -> OpaquePointer {
            let contextParams = createContextParams(from: options)
            guard let context = llama_init_from_model(model!, contextParams) else {
                throw LlamaLanguageModelError.contextInitializationFailed
            }
            guard llama_get_memory(context) != nil else {
                llama_free(context)
                throw LlamaLanguageModelError.encoderOnlyModel
            }
            llama_set_causal_attn(context, true)
            llama_set_n_threads(context, options.threads, options.threads)
            return context
        }

        /// Runs a chat text generation for the session, reusing the session's
        /// cached context when its state matches a prefix of the prompt.
        private func generateChatText(
            session: LanguageModelSession,
            prompt: String,
            maxTokens: Int,
            options: ResolvedGenerationOptions,
            onToken: (String) -> Bool
        ) throws -> LanguageModelSession.Usage {
            guard let model = self.model, let vocab = llama_model_get_vocab(model) else {
                throw LlamaLanguageModelError.contextInitializationFailed
            }

            let promptTokens = try tokenizeText(vocab: vocab, text: prompt)
            guard !promptTokens.isEmpty else {
                throw LlamaLanguageModelError.tokenizationFailed
            }

            if llama_model_has_encoder(model) {
                let context = try makeFreshContext(options: options)
                defer { llama_free(context) }
                return try performTokenGeneration(
                    context: context,
                    vocab: vocab,
                    promptTokens: promptTokens,
                    startIndex: 0,
                    maxTokens: maxTokens,
                    options: options,
                    onToken: onToken
                )
            }

            let (context, startIndex) = try acquireSessionContext(
                for: session,
                promptTokens: promptTokens,
                options: options
            )
            defer { releaseSessionContext(context) }
            llama_set_causal_attn(context, true)
            llama_set_n_threads(context, options.threads, options.threads)

            do {
                return try performTokenGeneration(
                    context: context,
                    vocab: vocab,
                    promptTokens: promptTokens,
                    startIndex: startIndex,
                    maxTokens: maxTokens,
                    options: options,
                    onToken: onToken
                )
            } catch {
                if isCachedSessionContext(context) {
                    discardCachedSessionContext()
                }
                throw error
            }
        }

        /// Creates a Llama language model.
        ///
        /// - Parameters:
        ///   - modelPath: The path to the GGUF model file.
        ///   - gpuLayers: The number of model layers to offload to the GPU.
        ///     Defaults to ``defaultGPULayerCount``.
        ///   - mmprojPath: The path to a multimodal projector GGUF file matching
        ///     the model. When provided, prompts may include image segments,
        ///     which are encoded through the projector. Defaults to `nil`
        ///     (text only).
        public init(
            modelPath: String,
            gpuLayers: Int32 = LlamaLanguageModel.defaultGPULayerCount,
            mmprojPath: String? = nil
        ) {
            self.modelPath = modelPath
            self.gpuLayers = gpuLayers
            self.mmprojPath = mmprojPath
            self.defaultOptions = ResolvedGenerationOptions()
        }

        /// 模型原生上下文窗口长度（读取 GGUF 自带 context_length，
        /// 经 llama_n_ctx_train 取得）。模型加载前为 nil。
        public var modelContextSize: UInt32? {
            nativeContextSize
        }

        deinit {
            if let cached = cachedSessionContext {
                llama_free(cached.context)
            }
            if let mtmdContext = mtmdContext {
                mtmd_free(mtmdContext)
            }
            if let model = model {
                llama_model_free(model)
            }
        }

        // MARK: - Tool calling

        /// Prompt-side tool state for one exchange: the detected syntax, the
        /// session's tool definitions, and the tool turns produced so far in
        /// the current resolve-and-continue loop.
        struct LlamaToolPromptContext {
            let format: LlamaToolCallFormat
            let definitions: [LlamaToolDefinition]
            var pendingEntries: [Transcript.Entry] = []

            init(format: LlamaToolCallFormat, tools: [any Tool]) throws {
                self.format = format
                self.definitions = try tools.filter(\.includesSchemaInInstructions).map { tool in
                    let schema = tool.parameters.withResolvedRoot() ?? tool.parameters
                    let data = try JSONEncoder().encode(schema)
                    let parameters = try JSONDecoder().decode(JSONValue.self, from: data).objectValue
                    return LlamaToolDefinition(
                        name: tool.name,
                        description: tool.description,
                        parameters: parameters
                    )
                }
            }
        }

        private struct ToolInvocationResult {
            let call: Transcript.ToolCall
            let output: Transcript.ToolOutput
        }

        private enum ToolResolutionOutcome {
            case stop(calls: [Transcript.ToolCall])
            case invocations([ToolInvocationResult])
        }

        private func currentToolCallFormat() -> LlamaToolCallFormat {
            guard let model = self.model else { return .hermesJSON }
            let template = llama_model_chat_template(model, nil).map { String(cString: $0) }
            return LlamaToolCallFormat.detect(template: template)
        }

        private func makeToolPromptContext(
            tools: [any Tool],
            pendingEntries: [Transcript.Entry] = []
        ) throws -> LlamaToolPromptContext? {
            guard (!tools.isEmpty || !pendingEntries.isEmpty), self.model != nil else { return nil }
            var context = try LlamaToolPromptContext(format: currentToolCallFormat(), tools: tools)
            context.pendingEntries = pendingEntries
            return context
        }

        private func makeTranscriptToolCalls(
            from parsedCalls: [LlamaParsedToolCall]
        ) throws -> [Transcript.ToolCall] {
            try parsedCalls.map { parsed in
                Transcript.ToolCall(
                    id: UUID().uuidString,
                    toolName: parsed.name,
                    arguments: try GeneratedContent(json: parsed.argumentsJSON)
                )
            }
        }

        private func resolveToolCalls(
            _ parsedCalls: [LlamaParsedToolCall],
            tools: [any Tool],
            session: LanguageModelSession
        ) async throws -> ToolResolutionOutcome {
            if parsedCalls.isEmpty { return .invocations([]) }

            var toolsByName: [String: any Tool] = [:]
            for tool in tools where toolsByName[tool.name] == nil {
                toolsByName[tool.name] = tool
            }

            let transcriptCalls = try makeTranscriptToolCalls(from: parsedCalls)

            if let delegate = session.toolExecutionDelegate {
                await delegate.didGenerateToolCalls(transcriptCalls, in: session)
            }

            var decisions: [ToolExecutionDecision] = []
            decisions.reserveCapacity(transcriptCalls.count)

            if let delegate = session.toolExecutionDelegate {
                for call in transcriptCalls {
                    let decision = await delegate.toolCallDecision(for: call, in: session)
                    if case .stop = decision {
                        return .stop(calls: transcriptCalls)
                    }
                    decisions.append(decision)
                }
            } else {
                decisions = Array(repeating: .execute, count: transcriptCalls.count)
            }

            var results: [ToolInvocationResult] = []
            results.reserveCapacity(transcriptCalls.count)

            for (index, call) in transcriptCalls.enumerated() {
                switch decisions[index] {
                case .stop:
                    return .stop(calls: transcriptCalls)
                case .provideOutput(let segments):
                    let output = Transcript.ToolOutput(
                        id: call.id,
                        toolName: call.toolName,
                        segments: segments
                    )
                    if let delegate = session.toolExecutionDelegate {
                        await delegate.didExecuteToolCall(call, output: output, in: session)
                    }
                    results.append(ToolInvocationResult(call: call, output: output))
                case .execute:
                    guard let tool = toolsByName[call.toolName] else {
                        let message = Transcript.Segment.text(.init(content: "Tool not found: \(call.toolName)"))
                        let output = Transcript.ToolOutput(
                            id: call.id,
                            toolName: call.toolName,
                            segments: [message]
                        )
                        if let delegate = session.toolExecutionDelegate {
                            await delegate.didExecuteToolCall(call, output: output, in: session)
                        }
                        results.append(ToolInvocationResult(call: call, output: output))
                        continue
                    }

                    do {
                        let segments = try await tool.makeOutputSegments(from: call.arguments)
                        let output = Transcript.ToolOutput(
                            id: call.id,
                            toolName: tool.name,
                            segments: segments
                        )
                        if let delegate = session.toolExecutionDelegate {
                            await delegate.didExecuteToolCall(call, output: output, in: session)
                        }
                        results.append(ToolInvocationResult(call: call, output: output))
                    } catch {
                        if let delegate = session.toolExecutionDelegate {
                            await delegate.didFailToolCall(call, error: error, in: session)
                        }
                        throw LanguageModelSession.ToolCallError(tool: tool, underlyingError: error)
                    }
                }
            }

            return .invocations(results)
        }

        public func respond<Content>(
            within session: LanguageModelSession,
            to prompt: Prompt,
            generating type: Content.Type,
            includeSchemaInPrompt: Bool,
            options: GenerationOptions
        ) async throws -> LanguageModelSession.Response<Content> where Content: Generable {
            try await respond(
                within: session,
                to: prompt,
                generating: type,
                schema: type.generationSchema,
                includeSchemaInPrompt: includeSchemaInPrompt,
                options: options
            )
        }

        public func respond(
            within session: LanguageModelSession,
            to prompt: Prompt,
            schema: GenerationSchema,
            includeSchemaInPrompt: Bool,
            options: GenerationOptions
        ) async throws -> LanguageModelSession.Response<GeneratedContent> {
            try await respond(
                within: session,
                to: prompt,
                generating: GeneratedContent.self,
                schema: schema,
                includeSchemaInPrompt: includeSchemaInPrompt,
                options: options
            )
        }

        private func respond<Content>(
            within session: LanguageModelSession,
            to prompt: Prompt,
            generating type: Content.Type,
            schema: GenerationSchema,
            includeSchemaInPrompt: Bool,
            options: GenerationOptions
        ) async throws -> LanguageModelSession.Response<Content> where Content: Generable {
            try ensureModelLoaded()

            let runtimeOptions = resolvedOptions(from: options)
            let structuredOptions = resolvedStructuredOptions(from: options)

            let imageMarker = mtmdContext != nil ? String(cString: mtmd_default_marker()) : nil

            if type == String.self {
                let maxTokens = Self.resolveMaxTokens(
                    requested: runtimeOptions.maximumResponseTokens,
                    contextSize: runtimeOptions.contextSize
                )
                let outputFormat = currentToolCallFormat()
                var toolRounds = ToolRoundLimit(provider: "Llama", options: options)
                var allEntries: [Transcript.Entry] = []
                var text = ""
                var usage = RequestsUsage()
                var pendingEntries: [Transcript.Entry] = []

                generationLoop: while true {
                    let requestContext = session.resolvedRequestContext()
                    if mmprojPath == nil {
                        try validateNoImageSegments(in: requestContext.transcript)
                    }
                    // llama.cpp has no tool-choice setting.
                    let tools = options.toolsOffered(from: requestContext.tools)
                    let toolContext = try makeToolPromptContext(
                        tools: tools,
                        pendingEntries: pendingEntries
                    )
                    var promptImages: [Data] = []
                    let fullPrompt = try formatPrompt(
                        requestContext: requestContext,
                        extraSystemMessage: nil,
                        assistantPrefill: runtimeOptions.assistantPrefill,
                        imageMarker: imageMarker,
                        images: &promptImages,
                        toolContext: toolContext
                    )

                    var accumulated = ""
                    let terminator = toolContext?.format.callTerminator
                    let collectToken: (String) -> Bool = { tokenText in
                        accumulated += tokenText
                        if let terminator,
                            accumulated.suffix(terminator.count + 8).contains(terminator)
                        {
                            return false
                        }
                        return true
                    }

                    let roundUsage: LanguageModelSession.Usage
                    if promptImages.isEmpty {
                        roundUsage = try generateChatText(
                            session: session,
                            prompt: fullPrompt,
                            maxTokens: maxTokens,
                            options: runtimeOptions,
                            onToken: collectToken
                        )
                    } else {
                        discardCachedSessionContext()
                        let context = try makeFreshContext(options: runtimeOptions)
                        defer { llama_free(context) }
                        roundUsage = try performMultimodalGeneration(
                            context: context,
                            prompt: fullPrompt,
                            images: promptImages,
                            maxTokens: maxTokens,
                            options: runtimeOptions,
                            onToken: collectToken
                        )
                    }

                    usage.add(roundUsage)

                    guard let format = toolContext?.format else {
                        text = outputFormat.streamingVisibleText(
                            in: accumulated,
                            withholdToolCalls: false,
                            holdPartialMarkers: false
                        )
                        break generationLoop
                    }
                    let (visibleText, parsedCalls) = format.parseToolCalls(in: accumulated)
                    // Keep the text from every round, as the streaming path does,
                    // so a preamble before a tool call is not lost.
                    text += visibleText
                    if parsedCalls.isEmpty {
                        break generationLoop
                    }

                    try toolRounds.record(completed: allEntries)

                    let resolution = try await resolveToolCalls(
                        parsedCalls,
                        tools: tools,
                        session: session
                    )
                    switch resolution {
                    case .stop(let calls):
                        if !calls.isEmpty {
                            allEntries.append(.toolCalls(Transcript.ToolCalls(calls)))
                        }
                        return LanguageModelSession.Response(
                            content: "" as! Content,
                            rawContent: GeneratedContent(""),
                            transcriptEntries: ArraySlice(allEntries),
                            requests: usage
                        )
                    case .invocations(let invocations):
                        guard !invocations.isEmpty else {
                            break generationLoop
                        }
                        let callsEntry = Transcript.Entry.toolCalls(
                            Transcript.ToolCalls(invocations.map(\.call))
                        )
                        allEntries.append(callsEntry)
                        pendingEntries.append(callsEntry)
                        for invocation in invocations {
                            let outputEntry = Transcript.Entry.toolOutput(invocation.output)
                            allEntries.append(outputEntry)
                            pendingEntries.append(outputEntry)
                        }
                    }
                }

                return LanguageModelSession.Response(
                    content: text as! Content,
                    rawContent: GeneratedContent(text),
                    transcriptEntries: ArraySlice(allEntries),
                    requests: usage
                )
            } else {
                let requestContext = session.resolvedRequestContext()
                if mmprojPath == nil {
                    try validateNoImageSegments(in: requestContext.transcript)
                }
                var promptImages: [Data] = []
                let fullPrompt: String
                if includeSchemaInPrompt {
                    fullPrompt = try formatPrompt(
                        requestContext: requestContext,
                        extraSystemMessage: schemaPrompt(for: schema),
                        assistantPrefill: runtimeOptions.assistantPrefill,
                        imageMarker: imageMarker,
                        images: &promptImages
                    )
                } else {
                    fullPrompt = try formatPrompt(
                        requestContext: requestContext,
                        extraSystemMessage: nil,
                        assistantPrefill: runtimeOptions.assistantPrefill,
                        imageMarker: imageMarker,
                        images: &promptImages
                    )
                }
                // Structured output does not go through the projector yet; refuse rather
                // than answer about an image the model never saw.
                guard promptImages.isEmpty else {
                    throw LlamaLanguageModelError.unsupportedFeature
                }
                let context = try makeFreshContext(options: runtimeOptions)
                defer { llama_free(context) }
                let maxTokens = Self.resolveMaxTokens(
                    requested: structuredOptions.maximumResponseTokens,
                    contextSize: structuredOptions.contextSize
                )
                let (jsonString, usage) = try await generateStructuredJSON(
                    context: context,
                    prompt: fullPrompt,
                    schema: schema,
                    maxTokens: maxTokens,
                    options: structuredOptions
                )
                let generatedContent = try GeneratedContent(json: jsonString)
                let content = try type.init(generatedContent)
                return LanguageModelSession.Response(
                    content: content,
                    rawContent: generatedContent,
                    transcriptEntries: ArraySlice([]),
                    usage: usage
                )
            }
        }

        public func streamResponse<Content>(
            within session: LanguageModelSession,
            to prompt: Prompt,
            generating type: Content.Type,
            includeSchemaInPrompt: Bool,
            options: GenerationOptions
        ) -> sending LanguageModelSession.ResponseStream<Content> where Content: Generable {
            streamResponse(
                within: session,
                to: prompt,
                generating: type,
                schema: type.generationSchema,
                includeSchemaInPrompt: includeSchemaInPrompt,
                options: options
            )
        }

        public func streamResponse(
            within session: LanguageModelSession,
            to prompt: Prompt,
            schema: GenerationSchema,
            includeSchemaInPrompt: Bool,
            options: GenerationOptions
        ) -> sending LanguageModelSession.ResponseStream<GeneratedContent> {
            streamResponse(
                within: session,
                to: prompt,
                generating: GeneratedContent.self,
                schema: schema,
                includeSchemaInPrompt: includeSchemaInPrompt,
                options: options
            )
        }

        private func streamResponse<Content>(
            within session: LanguageModelSession,
            to prompt: Prompt,
            generating type: Content.Type,
            schema: GenerationSchema,
            includeSchemaInPrompt: Bool,
            options: GenerationOptions
        ) -> sending LanguageModelSession.ResponseStream<Content> where Content: Generable {
            guard type == String.self else {
                return streamStructuredResponse {
                    try await self.respond(
                        within: session,
                        to: prompt,
                        generating: type,
                        schema: schema,
                        includeSchemaInPrompt: includeSchemaInPrompt,
                        options: options
                    )
                }
            }

            let stream: AsyncThrowingStream<LanguageModelSession.ResponseStream<Content>.Snapshot, any Error> =
                AsyncThrowingStream { continuation in
                    let task = Task {
                        do {
                            try ensureModelLoaded()

                            let runtimeOptions = resolvedOptions(from: options)
                            let maxTokens = Self.resolveMaxTokens(
                                requested: runtimeOptions.maximumResponseTokens,
                                contextSize: runtimeOptions.contextSize
                            )
                            let outputFormat = self.currentToolCallFormat()
                            var toolRounds = ToolRoundLimit(provider: "Llama", options: options)
                            var accumulatedEntries: [Transcript.Entry] = []
                            var emittedBase = ""
                            var usage = RequestsUsage()
                            var lastYieldedText: String?
                            var pendingEntries: [Transcript.Entry] = []
                            let imageMarker =
                                self.mtmdContext != nil ? String(cString: mtmd_default_marker()) : nil

                            func yieldSnapshot(_ text: String) {
                                lastYieldedText = text
                                let snapshot = LanguageModelSession.ResponseStream<Content>.Snapshot(
                                    content: (text as! Content).asPartiallyGenerated(),
                                    rawContent: GeneratedContent(text),
                                    transcriptEntries: ArraySlice(accumulatedEntries),
                                    requests: usage
                                )
                                continuation.yield(snapshot)
                            }

                            generationLoop: while true {
                                let requestContext = session.resolvedRequestContext()
                                if self.mmprojPath == nil {
                                    try self.validateNoImageSegments(in: requestContext.transcript)
                                }
                                // llama.cpp has no tool-choice setting.
                                let tools = options.toolsOffered(from: requestContext.tools)
                                let toolContext = try self.makeToolPromptContext(
                                    tools: tools,
                                    pendingEntries: pendingEntries
                                )
                                var promptImages: [Data] = []
                                let fullPrompt = try self.formatPrompt(
                                    requestContext: requestContext,
                                    extraSystemMessage: nil,
                                    assistantPrefill: runtimeOptions.assistantPrefill,
                                    imageMarker: imageMarker,
                                    images: &promptImages,
                                    toolContext: toolContext
                                )

                                var roundRaw = ""
                                let terminator = toolContext?.format.callTerminator
                                let withholdToolCalls = toolContext != nil
                                var thoughtParser = ThoughtParser(
                                    pairs: self.detectedThoughtPairs ?? ThoughtParser.defaultPairs
                                )
                                func updateReasoningEntry() {
                                    guard !thoughtParser.thinking.isEmpty else { return }
                                    let reasoningEntry = Transcript.Entry.reasoning(
                                        Transcript.Reasoning(
                                            origin: .llama,
                                            segments: [.text(
                                                Transcript.TextSegment(content: thoughtParser.thinking)
                                            )]
                                        )
                                    )
                                    if let lastIndex = accumulatedEntries.indices.last,
                                        case .reasoning = accumulatedEntries[lastIndex] {
                                        accumulatedEntries[lastIndex] = reasoningEntry
                                    } else {
                                        accumulatedEntries.append(reasoningEntry)
                                    }
                                }
                                let collectToken: (String) -> Bool = { tokenText in
                                    roundRaw += tokenText
                                    // Stream thinking spans as reasoning transcript entries,
                                    // separate from the visible body text.
                                    thoughtParser.consume(tokenText)
                                    updateReasoningEntry()
                                    let visible = outputFormat.streamingVisibleText(
                                        in: roundRaw,
                                        withholdToolCalls: withholdToolCalls,
                                        thoughtPairs: self.detectedThoughtPairs
                                    )
                                    let total = emittedBase + visible
                                    if !total.isEmpty, total != lastYieldedText {
                                        yieldSnapshot(total)
                                    }
                                    if let terminator,
                                        roundRaw.suffix(terminator.count + 8).contains(terminator)
                                    {
                                        return false
                                    }
                                    return true
                                }

                                let roundUsage: LanguageModelSession.Usage
                                if promptImages.isEmpty {
                                    roundUsage = try self.generateChatText(
                                        session: session,
                                        prompt: fullPrompt,
                                        maxTokens: maxTokens,
                                        options: runtimeOptions,
                                        onToken: collectToken
                                    )
                                } else {
                                    self.discardCachedSessionContext()
                                    let context = try self.makeFreshContext(options: runtimeOptions)
                                    defer { llama_free(context) }
                                    roundUsage = try self.performMultimodalGeneration(
                                        context: context,
                                        prompt: fullPrompt,
                                        images: promptImages,
                                        maxTokens: maxTokens,
                                        options: runtimeOptions,
                                        onToken: collectToken
                                    )
                                }

                                usage.add(roundUsage)
                                // Flush any unclosed thinking span into the reasoning entry.
                                thoughtParser.finalize()
                                updateReasoningEntry()
                                // Publish counts before tool execution, which may stop or fail.
                                yieldSnapshot(lastYieldedText ?? emittedBase)

                                if Task.isCancelled {
                                    break generationLoop
                                }

                                let roundVisible = outputFormat.streamingVisibleText(
                                    in: roundRaw,
                                    withholdToolCalls: withholdToolCalls,
                                    holdPartialMarkers: false,
                                    thoughtPairs: self.detectedThoughtPairs
                                )

                                guard let format = toolContext?.format else {
                                    emittedBase += roundVisible
                                    break generationLoop
                                }
                                let (_, parsedCalls) = format.parseToolCalls(in: roundRaw)
                                if parsedCalls.isEmpty {
                                    emittedBase += roundVisible
                                    break generationLoop
                                }

                                try toolRounds.record(completed: accumulatedEntries)

                                let resolution = try await self.resolveToolCalls(
                                    parsedCalls,
                                    tools: tools,
                                    session: session
                                )
                                switch resolution {
                                case .stop(let calls):
                                    emittedBase += roundVisible
                                    if !calls.isEmpty {
                                        accumulatedEntries.append(.toolCalls(Transcript.ToolCalls(calls)))
                                        yieldSnapshot(emittedBase)
                                    }
                                    break generationLoop
                                case .invocations(let invocations):
                                    guard !invocations.isEmpty else {
                                        emittedBase += roundVisible
                                        break generationLoop
                                    }
                                    let callsEntry = Transcript.Entry.toolCalls(
                                        Transcript.ToolCalls(invocations.map(\.call))
                                    )
                                    accumulatedEntries.append(callsEntry)
                                    pendingEntries.append(callsEntry)
                                    for invocation in invocations {
                                        let outputEntry = Transcript.Entry.toolOutput(invocation.output)
                                        accumulatedEntries.append(outputEntry)
                                        pendingEntries.append(outputEntry)
                                    }
                                    emittedBase += roundVisible
                                    yieldSnapshot(emittedBase)
                                }
                            }

                            if emittedBase != lastYieldedText {
                                yieldSnapshot(emittedBase)
                            }
                            continuation.finish()
                        } catch {
                            continuation.finish(throwing: error)
                        }
                    }

                    continuation.onTermination = { _ in
                        task.cancel()
                    }
                }

            return LanguageModelSession.ResponseStream(stream: stream)
        }

        // MARK: - Private Helpers

        private func ensureModelLoaded() throws {
            modelLoadLock.lock()
            defer { modelLoadLock.unlock() }

            guard !isModelLoaded else { return }

            // Check if model file exists
            guard FileManager.default.fileExists(atPath: modelPath) else {
                throw LlamaLanguageModelError.invalidModelPath
            }

            // 加载前内存预检：llama.cpp 在内存耗尽时会直接 abort()（进程级崩溃，
            // 无异常可捕获），这里提前估算并拦截，避免把整机内存打满导致系统卡死。
            try ensureSufficientMemoryForLoading()

            // Initialize backend lazily - must be done before loading model
            llama_backend_init()

            tokenCache = nil
            vocab = nil

            // Free any existing model before loading a new one
            discardCachedSessionContext()
            if let existingContext = mtmdContext {
                mtmd_free(existingContext)
                self.mtmdContext = nil
            }
            if let existingModel = model {
                llama_model_free(existingModel)
                self.model = nil
            }

            let modelParams = createModelParams()
            guard let loadedModel = llama_model_load_from_file(modelPath, modelParams) else {
                throw LlamaLanguageModelError.modelLoadFailed
            }

            if let mmprojPath {
                guard FileManager.default.fileExists(atPath: mmprojPath) else {
                    llama_model_free(loadedModel)
                    throw LlamaLanguageModelError.invalidModelPath
                }
                var mtmdParams = mtmd_context_params_default()
                mtmdParams.use_gpu = gpuLayers != 0
                mtmdParams.print_timings = false
                mtmdParams.n_threads = defaultOptions.threads
                guard let projector = mtmd_init_from_file(mmprojPath, loadedModel, mtmdParams) else {
                    llama_model_free(loadedModel)
                    throw LlamaLanguageModelError.modelLoadFailed
                }
                self.mtmdContext = projector
            }

            self.model = loadedModel
            self.vocab = llama_model_get_vocab(loadedModel)
            self.tokenCache = StructuredGenerationTokenCache()
            self.detectedThoughtPairs = ThoughtParser.detectThoughtSpans(model: loadedModel)
            // 模型原生训练上下文：GGUF 自带 context_length，
            // 未显式设置时用它作为上下文/输出上限的默认依据
            let native = llama_n_ctx_train(loadedModel)
            self.nativeContextSize = native > 0 ? UInt32(native) : nil
            self.isModelLoaded = true
        }

        /// 加载前内存预检：估算 GGUF 权重 + 计算缓冲所需内存，
        /// 可用内存不足时提前抛 `insufficientMemory`，避免 llama.cpp 内部分配失败
        /// 调用 abort() 直接终止进程（该路径无异常可捕获，会让 App/系统直接卡死）。
        private func ensureSufficientMemoryForLoading() throws {
            let fileSize: UInt64
            do {
                let attrs = try FileManager.default.attributesOfItem(atPath: modelPath)
                fileSize = (attrs[.size] as? NSNumber)?.uint64Value ?? 0
            } catch {
                throw LlamaLanguageModelError.invalidModelPath
            }
            guard fileSize > 0 else {
                throw LlamaLanguageModelError.invalidModelPath
            }

            #if os(macOS)
            // 权重 + 推理/计算缓冲保守按 1.5 倍估算；多模态投影器另计
            var estimate = Double(fileSize) * 1.5
            if let mmprojPath {
                if let attrs = try? FileManager.default.attributesOfItem(atPath: mmprojPath),
                   let m = (attrs[.size] as? NSNumber)?.uint64Value {
                    estimate += Double(m) * 1.2
                }
            }
            let available = Double(Self.availableMemoryBytes())
            // 记录估算基线供运行时护栏使用
            memoryLoadEstimateBytes = UInt64(estimate)
            // 保留 20% 余量，防止整机进入内存压力
            if estimate > available * 0.8 {
                throw LlamaLanguageModelError.insufficientMemory(
                    requiredBytes: UInt64(estimate),
                    availableBytes: UInt64(available)
                )
            }
            #endif
        }

        /// 运行时内存护栏：推理中 KV cache 随上下文持续增长、系统内存动态变化，
        /// 每 N 个 token 检查一次系统可用内存，低于阈值立即中断生成。
        /// 这避免 llama.cpp 在内存耗尽时调用 abort() 把整机干死机。
        private func checkRuntimeMemoryPressure() throws {
            #if os(macOS)
            let free = Self.availableMemoryBytes()
            // 阈值：模型估算的 1/4，保底 1GB；低于即认为内存吃紧并提前停止。
            // 留足余量，避免 macOS 内存压缩/换页导致整机卡顿（鼠标都动不了）。
            let lowerBound = max(memoryLoadEstimateBytes / 4, 1 * 1024 * 1024 * 1024)
            if free < lowerBound {
                throw LlamaLanguageModelError.insufficientMemory(
                    requiredBytes: memoryLoadEstimateBytes,
                    availableBytes: free
                )
            }
            #endif
        }

        /// 推荐推理线程数：取物理核心的一半（保底 2）。
        /// llama.cpp 每个线程占一个 CPU 核心全速运算，默认用全部核心会让
        /// 推理占满整机 CPU，UI 与系统其他应用全部卡顿。
        private static func recommendedThreadCount() -> Int32 {
            #if os(macOS)
            var physical = 0
            var size = MemoryLayout<Int>.size
            if sysctlbyname("hw.physicalcpu", &physical, &size, nil, 0) == 0, physical > 0 {
                return Int32(max(2, physical / 2))
            }
            #endif
            let logical = ProcessInfo.processInfo.processorCount
            return Int32(max(2, logical / 2))
        }

        /// macOS 当前可用内存（free + inactive 可回收页）。
        /// 注意：`os_proc_available_memory()` 是 iOS 专用 API，macOS 上不可用。
        #if os(macOS)
            private static func availableMemoryBytes() -> UInt64 {
                var stats = vm_statistics64()
                var count = mach_msg_type_number_t(
                    MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size
                )
                let result = withUnsafeMutablePointer(to: &stats) {
                    $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                        host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
                    }
                }
                guard result == KERN_SUCCESS else {
                    return ProcessInfo.processInfo.physicalMemory
                }
                let pageSize = UInt64(sysconf(Int32(_SC_PAGESIZE)))
                return (UInt64(stats.free_count) + UInt64(stats.inactive_count)) * pageSize
            }

            /// 本进程实际物理内存占用（phys_footprint，含 Metal/统一内存分配）。
            private static func processFootprintBytes() -> UInt64 {
                var info = task_vm_info_data_t()
                var count = mach_msg_type_number_t(
                    MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size
                )
                let kr = withUnsafeMutablePointer(to: &info) {
                    $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                        task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
                    }
                }
                guard kr == KERN_SUCCESS else { return 0 }
                return UInt64(info.phys_footprint)
            }

            /// 本进程实际 CPU 使用率（proc_pidinfo 的 user+system 差分，归一化到单核 0-100%）。
            /// 注意：host_cpu_load_info 的 idle tick 在 Apple Silicon 上不更新（恒显示 100%），
            /// 所以不用系统级采样，改用进程级 CPU 时间。
            private final class ProcessCPUSampleState: @unchecked Sendable {
                let lock = NSLock()
                var last: (cpuSeconds: Double, wall: CFAbsoluteTime)?
            }
            private static let processCPUSampleState = ProcessCPUSampleState()

            private static func processCPUPercent() -> Double {
                var info = proc_taskinfo()
                let size = MemoryLayout<proc_taskinfo>.size
                let r = proc_pidinfo(getpid(), PROC_PIDTASKINFO, 0, &info, Int32(size))
                guard r == size else { return 0 }
                // pti_total_user / pti_total_system 单位为微秒
                let cpuSeconds = Double(info.pti_total_user + info.pti_total_system) / 1_000_000
                processCPUSampleState.lock.lock()
                defer { processCPUSampleState.lock.unlock() }
                guard let last = processCPUSampleState.last, cpuSeconds >= last.cpuSeconds else {
                    processCPUSampleState.last = (cpuSeconds, CFAbsoluteTimeGetCurrent())
                    return 0
                }
                let now = CFAbsoluteTimeGetCurrent()
                let dt = max(now - last.wall, 0.05)
                let dCPU = cpuSeconds - last.cpuSeconds
                processCPUSampleState.last = (cpuSeconds, now)
                let ncpu = Double(ProcessInfo.processInfo.activeProcessorCount)
                let pct = dCPU / dt * 100 / ncpu
                return min(max(pct, 0), 100)
            }
        #endif

        /// 本地模型性能快照：权重/参数/上下文/KV cache/进程内存/系统可用/CPU/速度。
        public struct LlamaPerformanceSnapshot: Sendable {
            /// GGUF 模型权重字节数（llama_model_size）
            public let modelSizeBytes: UInt64
            /// 模型参数量
            public let parameterCount: UInt64
            /// 当前上下文已用 token 数
            public let contextTokens: UInt32
            /// KV cache 估算字节（context × 层 × 维度 × 4，f16）
            public let kvCacheBytes: UInt64
            /// 本进程实际物理内存占用（phys_footprint，统一内存含 GPU 分配）
            public let processFootprintBytes: UInt64
            /// 系统当前可用内存
            public let availableSystemBytes: UInt64
            /// 本进程 CPU 使用率（归一化到单核，0-100%）
            public let systemCPUPercent: Double
            /// 最近一次生成速度（token/秒）
            public let tokensPerSecond: Double
            /// Metal 卸载层数（-1 = 全部卸载到 GPU）
            public let gpuLayers: Int32
        }

        /// 本地模型性能快照（权重/参数/KV cache/进程内存/系统可用/CPU/速度）。
        /// 模型未加载或非 macOS 返回 nil。
        public func performanceSnapshot() -> LlamaPerformanceSnapshot? {
            guard isModelLoaded, let model else { return nil }
            #if os(macOS)
            let size = llama_model_size(model)
            let params = llama_model_n_params(model)
            let nLayer = Int32(llama_model_n_layer(model))
            let nEmbd = Int32(llama_model_n_embd(model))
            var ctxTokens: UInt32 = 0
            var kvBytes: UInt64 = 0
            sessionContextLock.lock()
            if let cached = cachedSessionContext?.context {
                ctxTokens = llama_n_ctx(cached)
                // KV cache ≈ context tokens × 层数 × 维度 × 2(K+V) × 2(f16)
                kvBytes = UInt64(ctxTokens) * UInt64(max(nLayer, 1)) * UInt64(max(nEmbd, 1)) * 4
            }
            sessionContextLock.unlock()
            statsLock.lock()
            let tps = lastTokensPerSecond
            statsLock.unlock()
            return LlamaPerformanceSnapshot(
                modelSizeBytes: size,
                parameterCount: params,
                contextTokens: ctxTokens,
                kvCacheBytes: kvBytes,
                processFootprintBytes: Self.processFootprintBytes(),
                availableSystemBytes: Self.availableMemoryBytes(),
                systemCPUPercent: Self.processCPUPercent(),
                tokensPerSecond: tps,
                gpuLayers: gpuLayers
            )
            #else
            return nil
            #endif
        }

        private func createModelParams() -> llama_model_params {
            var params = llama_model_default_params()
            params.n_gpu_layers = gpuLayers

            // Try to reduce memory usage
            params.load_mode = LLAMA_LOAD_MODE_MMAP
            return params
        }

        private func resolvedOptions(from options: GenerationOptions) -> ResolvedGenerationOptions {
            var base = defaultOptions
            if let temp = options.temperature {
                base.temperature = Float(temp)
            }
            // 模型原生 context 自动接管默认（用户显式设置仍优先）
            if let native = nativeContextSize, native > 0 {
                base.contextSize = native
            }

            return ResolvedGenerationOptions(
                base: base,
                overrides: options[custom: LlamaLanguageModel.self],
                sampling: options.sampling,
                maximumResponseTokens: options.maximumResponseTokens
            )
        }

        /// Builds structured-generation defaults while honoring explicit overrides.
        private func resolvedStructuredOptions(from options: GenerationOptions) -> ResolvedGenerationOptions {
            var base = defaultOptions
            if let temp = options.temperature {
                base.temperature = Float(temp)
            } else {
                base.temperature = 0.2
            }
            base.topP = 0.95
            base.repeatPenalty = 1.1
            base.repeatLastN = 64
            // 模型原生 context 自动接管默认（用户显式设置仍优先）
            if let native = nativeContextSize, native > 0 {
                base.contextSize = native
            }

            return ResolvedGenerationOptions(
                base: base,
                overrides: options[custom: LlamaLanguageModel.self],
                sampling: options.sampling,
                maximumResponseTokens: options.maximumResponseTokens
            )
        }

        private func createContextParams(from options: ResolvedGenerationOptions) -> llama_context_params {
            var params = llama_context_default_params()
            params.n_ctx = options.contextSize
            params.n_batch = options.batchSize
            params.n_threads = options.threads
            params.n_threads_batch = options.threads
            return params
        }

        private func applySampling(
            sampler: UnsafeMutablePointer<llama_sampler>,
            effectiveTemperature: Float,
            options: ResolvedGenerationOptions
        ) {
            if let mirostat = options.mirostat {
                llama_sampler_chain_add(sampler, llama_sampler_init_temp(effectiveTemperature))

                switch mirostat {
                case .v1(let tau, let eta):
                    llama_sampler_chain_add(
                        sampler,
                        llama_sampler_init_mirostat(
                            Int32(options.contextSize),
                            options.seed,
                            tau,
                            eta,
                            100
                        )
                    )
                case .v2(let tau, let eta):
                    llama_sampler_chain_add(sampler, llama_sampler_init_mirostat_v2(options.seed, tau, eta))
                }
                return
            }

            if let sampling = options.sampling {
                switch sampling.mode {
                case .greedy:
                    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(1))
                    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(1.0, 1))
                    llama_sampler_chain_add(sampler, llama_sampler_init_greedy())
                case .topK(let k, let seed):
                    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(Int32(k)))
                    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(1.0, 1))
                    llama_sampler_chain_add(sampler, llama_sampler_init_temp(effectiveTemperature))
                    let samplingSeed = seed.map(UInt32.init) ?? options.seed
                    llama_sampler_chain_add(sampler, llama_sampler_init_dist(samplingSeed))
                case .nucleus(let threshold, let seed):
                    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(0))
                    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(Float(threshold), 1))
                    llama_sampler_chain_add(sampler, llama_sampler_init_temp(effectiveTemperature))
                    let samplingSeed = seed.map(UInt32.init) ?? options.seed
                    llama_sampler_chain_add(sampler, llama_sampler_init_dist(samplingSeed))
                }
                return
            }

            if options.topK > 0 {
                llama_sampler_chain_add(sampler, llama_sampler_init_top_k(options.topK))
            }
            if options.topP < 1.0 {
                llama_sampler_chain_add(sampler, llama_sampler_init_top_p(options.topP, 1))
            }
            llama_sampler_chain_add(sampler, llama_sampler_init_temp(effectiveTemperature))
            llama_sampler_chain_add(sampler, llama_sampler_init_dist(options.seed))
        }

        /// Builds a JSONSchema-informed prompt for structured output.
        private func schemaPrompt(for schema: GenerationSchema) -> String {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            guard
                let data = try? encoder.encode(schema),
                let jsonSchema = try? JSONDecoder().decode(JSONSchema.self, from: data),
                let schemaJSON = String(data: data, encoding: .utf8)
            else {
                return schema.schemaPrompt()
            }

            var header = "Respond with valid JSON matching this \(jsonSchema.typeName) schema"
            if let description = jsonSchema.description, !description.isEmpty {
                header += " (\(description))"
            }

            if let constValue = jsonSchema.const,
                let data = try? encoder.encode(constValue),
                let constString = String(data: data, encoding: .utf8)
            {
                header += ". Expected value: \(constString)"
            } else if let enumValues = jsonSchema.enum, !enumValues.isEmpty,
                let data = try? encoder.encode(enumValues),
                let enumString = String(data: data, encoding: .utf8)
            {
                header += ". Allowed values: \(enumString)"
            }

            return "\(header):\n\(schemaJSON)"
        }

        // MARK: - Structured JSON Generation

        private func generateStructuredJSON(
            context: OpaquePointer,
            prompt: String,
            schema: GenerationSchema,
            maxTokens: Int,
            options: ResolvedGenerationOptions
        ) async throws -> (String, LanguageModelSession.Usage) {
            // 推理串行门（async 安全）：结构化生成与文本生成共享同一模型/tokenCache，
            // 经 actor 串行执行，避免并发访问 tokenCache / 竞争模型状态。
            await structuredGate.acquire()
            do {
                let result = try await self.performStructuredGeneration(
                    context: context,
                    prompt: prompt,
                    schema: schema,
                    maxTokens: maxTokens,
                    options: options
                )
                await structuredGate.release()
                return result
            } catch {
                await structuredGate.release()
                throw error
            }
        }

        private func performStructuredGeneration(
            context: OpaquePointer,
            prompt: String,
            schema: GenerationSchema,
            maxTokens: Int,
            options: ResolvedGenerationOptions
        ) async throws -> (String, LanguageModelSession.Usage) {
            guard let vocab = llama_model_get_vocab(model!) else {
                throw LlamaLanguageModelError.contextInitializationFailed
            }

            let promptTokens = try tokenizeText(vocab: vocab, text: prompt)
            guard !promptTokens.isEmpty else {
                throw LlamaLanguageModelError.tokenizationFailed
            }

            let batchPointer = UnsafeMutablePointer<llama_batch>.allocate(capacity: 1)
            batchPointer.initialize(to: llama_batch_init(Int32(options.batchSize), 0, 1))
            defer {
                llama_batch_free(batchPointer.pointee)
                batchPointer.deinitialize(count: 1)
                batchPointer.deallocate()
            }

            let hasEncoder = try prepareInitialBatch(
                batch: &batchPointer.pointee,
                promptTokens: promptTokens,
                model: model!,
                vocab: vocab,
                context: context,
                batchSize: options.batchSize,
                contextSize: options.contextSize
            )

            guard let sampler = llama_sampler_chain_init(llama_sampler_chain_default_params()) else {
                throw LlamaLanguageModelError.decodingFailed
            }
            defer { llama_sampler_free(sampler) }
            let samplerPointer = UnsafeMutablePointer<llama_sampler>(sampler)

            if options.repeatPenalty != 1.0 || options.frequencyPenalty != 0.0 || options.presencePenalty != 0.0 {
                llama_sampler_chain_add(
                    samplerPointer,
                    llama_sampler_init_penalties(
                        llama_vocab_n_tokens(vocab),
                        options.repeatLastN,
                        options.repeatPenalty,
                        options.frequencyPenalty,
                        options.presencePenalty
                    )
                )
            }
            applySampling(sampler: samplerPointer, effectiveTemperature: options.temperature, options: options)

            let vocabSize = Int(llama_vocab_n_tokens(vocab))
            let initialPosition: Int32 = hasEncoder ? 1 : Int32(promptTokens.count)

            let backend = LlamaTokenBackend(
                context: context,
                vocab: vocab,
                vocabSize: vocabSize,
                sampler: samplerPointer,
                batch: batchPointer,
                position: initialPosition,
                maximumTokens: maxTokens,
                endTokens: [],
                tokenToTextFn: { [self] token in self.tokenToText(vocab: vocab, token: llama_token(token)) },
                memoryCheckFn: { [self] in try self.checkRuntimeMemoryPressure() }
            )
            var generator = try ConstrainedJSONGenerator(backend: backend, schema: schema, tokenCache: tokenCache)
            let json = try await generator.generate()
            return (
                json,
                LocalGenerationUsage(
                    promptTokenCount: promptTokens.count,
                    generatedTokenCount: generator.generatedTokenCount
                ).value
            )
        }

        private struct LlamaTokenBackend: TokenBackend {
            let context: OpaquePointer
            let vocab: OpaquePointer
            let vocabSize: Int
            let sampler: UnsafeMutablePointer<llama_sampler>
            let batch: UnsafeMutablePointer<llama_batch>
            let tokenToTextFn: (Int) -> String?
            let tokensExcludedFromRepetitionPenalty: Set<Int>
            let endTokens: Set<Int>
            /// 运行时内存护栏（结构化生成路径每 64 token 调用一次）
            let memoryCheckFn: (() throws -> Void)?
            /// decode 调用计数（用于按频率触发内存检查）
            private var decodeCount = 0

            var position: Int32
            var remainingTokens: Int
            let totalTokenBudget: Int
            let eosToken: Int

            init(
                context: OpaquePointer,
                vocab: OpaquePointer,
                vocabSize: Int,
                sampler: UnsafeMutablePointer<llama_sampler>,
                batch: UnsafeMutablePointer<llama_batch>,
                position: Int32,
                maximumTokens: Int,
                endTokens: Set<Int>? = nil,
                tokenToTextFn: @escaping (Int) -> String?,
                memoryCheckFn: (() throws -> Void)? = nil
            ) {
                self.context = context
                self.vocab = vocab
                self.vocabSize = vocabSize
                self.sampler = sampler
                self.batch = batch
                self.position = position
                self.remainingTokens = maximumTokens
                self.totalTokenBudget = maximumTokens
                self.eosToken = Int(llama_vocab_eos(vocab))
                self.memoryCheckFn = memoryCheckFn

                if let endTokens {
                    self.endTokens = endTokens
                } else {
                    let eotTokenValue = llama_vocab_eot(vocab)
                    let endOfTurnToken = eotTokenValue != LLAMA_TOKEN_NULL ? Int(eotTokenValue) : eosToken
                    self.endTokens = [self.eosToken, endOfTurnToken]
                }

                self.tokenToTextFn = tokenToTextFn
                self.tokensExcludedFromRepetitionPenalty = Self.buildTokensExcludedFromRepetitionPenalty(
                    vocabSize: vocabSize,
                    tokenToText: tokenToTextFn
                )
            }

            func isSpecialToken(_ token: Int) -> Bool {
                let attributes = llama_vocab_get_attr(vocab, llama_token(token))
                return (attributes.rawValue & LLAMA_TOKEN_ATTR_CONTROL.rawValue) != 0
            }

            private static func buildTokensExcludedFromRepetitionPenalty(
                vocabSize: Int,
                tokenToText: (Int) -> String?
            ) -> Set<Int> {
                let excludedTexts: Set<String> = ["{", "}", "[", "]", ",", ":", "\""]
                var excluded = Set<Int>()
                excluded.reserveCapacity(excludedTexts.count * 4)

                for token in 0 ..< vocabSize {
                    guard let text = tokenToText(token) else { continue }
                    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
                    if excludedTexts.contains(trimmed) {
                        excluded.insert(token)
                    }
                }

                return excluded
            }

            func tokenize(_ text: String) throws -> [Int] {
                let utf8Count = text.utf8.count
                let capacity = Int32(max(utf8Count * 2, 8))
                let tokens = UnsafeMutablePointer<llama_token>.allocate(capacity: Int(capacity))
                defer { tokens.deallocate() }

                let tokenCount = llama_tokenize(
                    vocab,
                    text,
                    Int32(utf8Count),
                    tokens,
                    capacity,
                    false,
                    false
                )
                guard tokenCount > 0 else { return [] }
                return Array(UnsafeBufferPointer(start: tokens, count: Int(tokenCount))).map { Int($0) }
            }

            func tokenText(_ token: Int) -> String? {
                tokenToTextFn(token)
            }

            mutating func decode(_ token: Int) async throws {
                decodeCount += 1
                // 结构化生成路径的运行时内存护栏（每 64 token 检查一次）
                if decodeCount % 64 == 0, let memoryCheckFn {
                    try memoryCheckFn()
                }

                let llamaToken = llama_token(token)

                batch.pointee.n_tokens = 1
                batch.pointee.token[0] = llamaToken
                batch.pointee.pos[0] = position
                batch.pointee.n_seq_id[0] = 1
                if let seqIds = batch.pointee.seq_id, let seqId = seqIds[0] {
                    seqId[0] = 0
                }
                batch.pointee.logits[0] = 1

                position += 1
                remainingTokens -= 1

                let decodeResult = llama_decode(context, batch.pointee)
                guard decodeResult == 0 else {
                    throw LlamaLanguageModelError.decodingFailed
                }

                if !tokensExcludedFromRepetitionPenalty.contains(Int(llamaToken)) {
                    llama_sampler_accept(sampler, llamaToken)
                }
            }

            mutating func sample(from allowedTokens: Set<Int>) async throws -> Int {
                // Masking llama_get_logits in place does not constrain
                // llama_sampler_sample: the context can refresh that buffer when
                // the sampler fetches logits, dropping the mask. Build a candidate
                // array holding only the allowed tokens and run the sampler chain
                // over it instead.
                guard let logits = llama_get_logits_ith(context, batch.pointee.n_tokens - 1) else {
                    return eosToken
                }

                var candidates = allowedTokens.compactMap { token -> llama_token_data? in
                    guard token >= 0, token < vocabSize else { return nil }
                    return llama_token_data(id: llama_token(token), logit: logits[token], p: 0)
                }
                guard !candidates.isEmpty else {
                    return eosToken
                }

                let selectedToken = candidates.withUnsafeMutableBufferPointer { buffer -> Int in
                    var array = llama_token_data_array(
                        data: buffer.baseAddress,
                        size: buffer.count,
                        selected: -1,
                        sorted: false
                    )
                    llama_sampler_apply(sampler, &array)
                    guard array.selected >= 0, array.selected < Int64(array.size), let data = array.data else {
                        return eosToken
                    }
                    return Int(data[Int(array.selected)].id)
                }
                return selectedToken
            }
        }

        private func performTokenGeneration(
            context: OpaquePointer,
            vocab: OpaquePointer,
            promptTokens: [llama_token],
            startIndex: Int,
            maxTokens: Int,
            options: ResolvedGenerationOptions,
            onToken: (String) -> Bool
        ) throws -> LanguageModelSession.Usage {
            // 推理串行门：同一实例同时只允许一个生成，避免共享 context/batch 竞争
            inferenceGate.wait()
            defer { inferenceGate.signal() }

            let generationStart = Date()
            var generatedTokenCount = 0
            defer {
                statsLock.lock()
                let elapsed = Date().timeIntervalSince(generationStart)
                lastTokensPerSecond = elapsed > 0 ? Double(generatedTokenCount) / elapsed : 0
                statsLock.unlock()
            }

            guard let model = self.model else {
                throw LlamaLanguageModelError.modelLoadFailed
            }

            sessionContextLock.lock()
            lastReusedTokenCount = startIndex
            lastPrefillTokenCount = promptTokens.count - startIndex
            sessionContextLock.unlock()

            // Initialize batch
            var batch = llama_batch_init(Int32(options.batchSize), 0, 1)
            defer { llama_batch_free(batch) }

            let hasEncoder = try prepareInitialBatch(
                batch: &batch,
                promptTokens: promptTokens,
                model: model,
                vocab: vocab,
                context: context,
                batchSize: options.batchSize,
                contextSize: options.contextSize,
                startIndex: startIndex
            )

            // Initialize sampler chain with options
            guard let sampler = llama_sampler_chain_init(llama_sampler_chain_default_params()) else {
                throw LlamaLanguageModelError.decodingFailed
            }
            defer { llama_sampler_free(sampler) }
            let samplerPtr = UnsafeMutablePointer<llama_sampler>(sampler)

            let effectiveTemperature = Float(options.temperature)

            // Apply repeat/frequency/presence penalties from custom options
            let effectiveRepeatPenalty = options.repeatPenalty
            let effectiveRepeatLastN = options.repeatLastN
            let effectiveFrequencyPenalty = options.frequencyPenalty
            let effectivePresencePenalty = options.presencePenalty

            if effectiveRepeatPenalty != 1.0 || effectiveFrequencyPenalty != 0.0 || effectivePresencePenalty != 0.0 {
                llama_sampler_chain_add(
                    samplerPtr,
                    llama_sampler_init_penalties(
                        llama_vocab_n_tokens(vocab),
                        effectiveRepeatLastN,
                        effectiveRepeatPenalty,
                        effectiveFrequencyPenalty,
                        effectivePresencePenalty
                    )
                )
            }

            // Check for mirostat sampling (takes precedence over standard sampling)
            applySampling(sampler: samplerPtr, effectiveTemperature: effectiveTemperature, options: options)

            // Generate tokens one by one
            // Track position - for encoder-decoder models, we start from position 1 (after decoder start token)
            // For decoder-only models, we continue from the end of the prompt
            var n_cur: Int32 = hasEncoder ? 1 : Int32(promptTokens.count)
            var counts = LocalGenerationUsage(
                promptTokenCount: promptTokens.count,
                cachedTokenCount: startIndex
            )
            var decodedTokens = promptTokens

            for tokenIndex in 0 ..< maxTokens {
                // 运行时内存护栏：KV cache 随上下文增长，内存动态变化，
                // 每 64 token 检查一次系统可用内存，吃紧即中断（而非等 abort 干死机）
                if tokenIndex % 64 == 0 {
                    try checkRuntimeMemoryPressure()
                }
                if Task.isCancelled {
                    break
                }

                // Sample next token from logits of the last token we just decoded
                let nextToken = llama_sampler_sample(sampler, context, batch.n_tokens - 1)
                llama_sampler_accept(sampler, nextToken)

                // Check for end of sequence
                if !counts.acceptToken(isEndOfGeneration: llama_vocab_is_eog(vocab, nextToken)) {
                    break
                }

                // Convert token to text and yield it
                if let tokenText = tokenToText(vocab: vocab, token: nextToken) {
                    // EOG / 会话结构标签兜底（GGUF 缺 eos_token_id 时
                    // llama_vocab_is_eog 失效，按文本拦截）
                    if isTerminalTokenText(tokenText) {
                        break
                    }
                    guard onToken(tokenText) else {
                        break
                    }
                }

                // Prepare batch for next token
                batch.n_tokens = 1
                batch.token[0] = nextToken
                batch.pos[0] = n_cur
                batch.n_seq_id[0] = 1
                if let seq_ids = batch.seq_id, let seq_id = seq_ids[0] {
                    seq_id[0] = 0
                }
                batch.logits[0] = 1

                n_cur += 1

                let decodeResult = llama_decode(context, batch)
                guard decodeResult == 0 else {
                    break
                }
                decodedTokens.append(nextToken)
                generatedTokenCount += 1
            }

            recordCachedTokens(decodedTokens, context: context)
            return counts.value
        }

        /// Count native chunks, including images; their positional spans are not token counts.
        internal static func multimodalPromptTokenCount(
            chunkCount: Int,
            tokenCountAtIndex: (Int) -> Int
        ) -> Int {
            (0 ..< chunkCount).reduce(0) { $0 + tokenCountAtIndex($1) }
        }

        /// Evaluates a marker-annotated multimodal prompt through the projector,
        /// then generates text tokens from the resulting state.
        private func performMultimodalGeneration(
            context: OpaquePointer,
            prompt: String,
            images: [Data],
            maxTokens: Int,
            options: ResolvedGenerationOptions,
            onToken: (String) -> Bool
        ) throws -> LanguageModelSession.Usage {
            // 推理串行门：多模态生成与文本/JSON 生成共享同一模型实例，必须串行
            inferenceGate.wait()
            defer { inferenceGate.signal() }

            guard let mtmdContext, let model = self.model,
                let vocab = llama_model_get_vocab(model)
            else {
                throw LlamaLanguageModelError.contextInitializationFailed
            }

            var promptTokenCount = 0
            var pastPosition: llama_pos = 0
            try projectorLocked {
                var bitmaps: [OpaquePointer?] = []
                defer {
                    for bitmap in bitmaps {
                        if let bitmap {
                            mtmd_bitmap_free(bitmap)
                        }
                    }
                }
                for imageData in images {
                    // Pinned to the current llama.swift signature. llama.cpp master adds a
                    // trailing options argument to this helper; update alongside the dependency.
                    let wrapper = imageData.withUnsafeBytes { raw -> mtmd_helper_bitmap_wrapper in
                        mtmd_helper_bitmap_init_from_buf(
                            mtmdContext,
                            raw.bindMemory(to: UInt8.self).baseAddress,
                            imageData.count,
                            false
                        )
                    }
                    if let videoContext = wrapper.video_ctx {
                        if let bitmap = wrapper.bitmap {
                            mtmd_bitmap_free(bitmap)
                        }
                        mtmd_helper_video_free(videoContext)
                        throw LlamaLanguageModelError.unsupportedFeature
                    }
                    guard let bitmap = wrapper.bitmap else {
                        throw LlamaLanguageModelError.encodingFailed
                    }
                    bitmaps.append(bitmap)
                }

                guard let chunks = mtmd_input_chunks_init() else {
                    throw LlamaLanguageModelError.encodingFailed
                }
                defer { mtmd_input_chunks_free(chunks) }

                let tokenizeResult = prompt.withCString { cPrompt -> Int32 in
                    var inputText = mtmd_input_text(
                        text: cPrompt,
                        text_len: strlen(cPrompt),
                        add_special: true,
                        parse_special: true
                    )
                    return bitmaps.withUnsafeMutableBufferPointer { buffer in
                        mtmd_tokenize(mtmdContext, chunks, &inputText, buffer.baseAddress, buffer.count)
                    }
                }
                guard tokenizeResult == 0 else {
                    throw LlamaLanguageModelError.tokenizationFailed
                }

                // Multimodal token counts can differ from positional offsets (e.g. M-RoPE).
                promptTokenCount = Self.multimodalPromptTokenCount(chunkCount: mtmd_input_chunks_size(chunks)) {
                    Int(mtmd_input_chunk_get_n_tokens(mtmd_input_chunks_get(chunks, $0)))
                }

                let evalResult = mtmd_helper_eval_chunks(
                    mtmdContext,
                    context,
                    chunks,
                    0,
                    0,
                    Int32(options.batchSize),
                    true,
                    &pastPosition
                )
                guard evalResult == 0 else {
                    throw LlamaLanguageModelError.decodingFailed
                }
            }

            guard let sampler = llama_sampler_chain_init(llama_sampler_chain_default_params()) else {
                throw LlamaLanguageModelError.decodingFailed
            }
            defer { llama_sampler_free(sampler) }
            let samplerPtr = UnsafeMutablePointer<llama_sampler>(sampler)

            if options.repeatPenalty != 1.0 || options.frequencyPenalty != 0.0 || options.presencePenalty != 0.0 {
                llama_sampler_chain_add(
                    samplerPtr,
                    llama_sampler_init_penalties(
                        llama_vocab_n_tokens(vocab),
                        options.repeatLastN,
                        options.repeatPenalty,
                        options.frequencyPenalty,
                        options.presencePenalty
                    )
                )
            }
            applySampling(sampler: samplerPtr, effectiveTemperature: options.temperature, options: options)

            var batch = llama_batch_init(1, 0, 1)
            defer { llama_batch_free(batch) }

            var n_cur: Int32 = Int32(pastPosition)
            var sampleIndex: Int32 = -1
            var counts = LocalGenerationUsage(promptTokenCount: promptTokenCount)

            for _ in 0 ..< maxTokens {
                if Task.isCancelled {
                    break
                }

                let nextToken = llama_sampler_sample(samplerPtr, context, sampleIndex)
                llama_sampler_accept(samplerPtr, nextToken)

                if !counts.acceptToken(isEndOfGeneration: llama_vocab_is_eog(vocab, nextToken)) {
                    break
                }

                if let tokenText = tokenToText(vocab: vocab, token: nextToken) {
                    // EOG / 会话结构标签兜底（GGUF 缺 eos_token_id 时
                    // llama_vocab_is_eog 失效，按文本拦截）
                    if isTerminalTokenText(tokenText) {
                        break
                    }
                    guard onToken(tokenText) else {
                        break
                    }
                }

                batch.n_tokens = 1
                batch.token[0] = nextToken
                batch.pos[0] = n_cur
                batch.n_seq_id[0] = 1
                if let seq_ids = batch.seq_id, let seq_id = seq_ids[0] {
                    seq_id[0] = 0
                }
                batch.logits[0] = 1

                n_cur += 1

                guard llama_decode(context, batch) == 0 else {
                    break
                }
                sampleIndex = 0
            }
            return counts.value
        }

        // MARK: - Image Validation

        private func validateNoImageSegments(in transcript: Transcript) throws {
            // Check for image segments in the most recent prompt from the transcript
            for entry in transcript.reversed() {
                if case .prompt(let p) = entry {
                    for segment in p.segments {
                        if case .image = segment {
                            throw LlamaLanguageModelError.unsupportedFeature
                        }
                    }
                    break
                }
            }
        }

        // MARK: - Helper Methods

        /// Prepares the initial batch for text generation, handling encoder-decoder vs decoder-only models.
        ///
        /// Decoder-only prompts longer than the batch capacity are ingested in
        /// batch-sized chunks. Encoder models must fit the prompt in one batch.
        ///
        /// - Parameters:
        ///   - batch: The batch to prepare (must be initialized with sufficient capacity).
        ///   - promptTokens: The tokenized prompt tokens.
        ///   - model: The loaded model.
        ///   - vocab: The model vocabulary.
        ///   - context: The model context.
        ///   - batchSize: The batch capacity per decode call.
        ///   - contextSize: The context window the prompt must fit within.
        ///   - startIndex: The index of the first prompt token to decode. Earlier
        ///     tokens are already present in the context's state. Defaults to `0`.
        /// - Returns: `true` if the model has an encoder (for position tracking during generation).
        /// - Throws: `promptExceedsContextWindow` if the prompt cannot fit in the context window,
        ///   `insufficientMemory` if an encoder prompt exceeds the batch capacity, `encoderOnlyModel`
        ///   if the model cannot generate text, `encodingFailed` or `decodingFailed` on failure.
        private func prepareInitialBatch(
            batch: inout llama_batch,
            promptTokens: [llama_token],
            model: OpaquePointer,
            vocab: OpaquePointer,
            context: OpaquePointer,
            batchSize: UInt32,
            contextSize: UInt32,
            startIndex: Int = 0
        ) throws -> Bool {
            // Leave at least one context cell free for generation.
            guard promptTokens.count < contextSize else {
                throw LlamaLanguageModelError.promptExceedsContextWindow
            }

            let hasEncoder = llama_model_has_encoder(model)
            let hasDecoder = llama_model_has_decoder(model)

            // Encoder models ingest the full prompt in a single llama_encode call.
            guard !hasEncoder || (startIndex == 0 && promptTokens.count <= batchSize) else {
                throw LlamaLanguageModelError.insufficientMemory(requiredBytes: 0, availableBytes: 0)
            }

            if hasEncoder {
                // For encoder models, first encode the prompt
                batch.n_tokens = Int32(promptTokens.count)
                for i in 0 ..< promptTokens.count {
                    let idx = Int(i)
                    batch.token[idx] = promptTokens[idx]
                    batch.pos[idx] = Int32(i)
                    batch.n_seq_id[idx] = 1
                    if let seq_ids = batch.seq_id, let seq_id = seq_ids[idx] {
                        seq_id[0] = 0
                    }
                    batch.logits[idx] = 0
                }

                guard llama_encode(context, batch) == 0 else {
                    throw LlamaLanguageModelError.encodingFailed
                }

                if hasDecoder {
                    // For encoder-decoder models, start decoding with decoder start token
                    var decoderStartToken = llama_model_decoder_start_token(model)
                    if decoderStartToken == LLAMA_TOKEN_NULL {
                        decoderStartToken = llama_vocab_bos(vocab)
                    }

                    batch.n_tokens = 1
                    batch.token[0] = decoderStartToken
                    batch.pos[0] = 0
                    batch.n_seq_id[0] = 1
                    if let seq_ids = batch.seq_id, let seq_id = seq_ids[0] {
                        seq_id[0] = 0
                    }
                    batch.logits[0] = 1

                    guard llama_decode(context, batch) == 0 else {
                        throw LlamaLanguageModelError.decodingFailed
                    }
                } else {
                    // Encoder-only model (like BERT) - cannot generate text.
                    // This architectural check complements the earlier KV cache check,
                    // catching models by their architecture type.
                    throw LlamaLanguageModelError.encoderOnlyModel
                }
            } else {
                // Standard decoder-only model (most LLMs): feed the prompt in
                // batch-sized chunks with absolute positions, requesting logits
                // only for the final token.
                let capacity = Int(batchSize)
                var start = startIndex
                while start < promptTokens.count {
                    let count = min(capacity, promptTokens.count - start)
                    batch.n_tokens = Int32(count)
                    for i in 0 ..< count {
                        batch.token[i] = promptTokens[start + i]
                        batch.pos[i] = Int32(start + i)
                        batch.n_seq_id[i] = 1
                        if let seq_ids = batch.seq_id, let seq_id = seq_ids[i] {
                            seq_id[0] = 0
                        }
                        batch.logits[i] = 0
                    }

                    if start + count == promptTokens.count {
                        batch.logits[count - 1] = 1
                    }

                    guard llama_decode(context, batch) == 0 else {
                        throw LlamaLanguageModelError.decodingFailed
                    }
                    start += count
                }
            }

            return hasEncoder
        }

        private func formatPrompt(
            requestContext: LanguageModelSession.RequestContext,
            extraSystemMessage: String? = nil,
            assistantPrefill: String? = nil,
            toolContext: LlamaToolPromptContext? = nil
        ) throws -> String {
            var images: [Data] = []
            return try formatPrompt(
                requestContext: requestContext,
                extraSystemMessage: extraSystemMessage,
                assistantPrefill: assistantPrefill,
                imageMarker: nil,
                images: &images,
                toolContext: toolContext
            )
        }

        private func formatPrompt(
            requestContext: LanguageModelSession.RequestContext,
            extraSystemMessage: String?,
            assistantPrefill: String?,
            imageMarker: String?,
            images: inout [Data],
            toolContext: LlamaToolPromptContext? = nil
        ) throws -> String {
            guard let model = self.model else {
                throw LlamaLanguageModelError.modelLoadFailed
            }

            var messages: [(role: String, content: String)] = []

            func appendEntry(_ entry: Transcript.Entry) throws {
                switch entry {
                case .instructions(let instructions):
                    let text = try extractContent(
                        from: instructions.segments,
                        imageMarker: imageMarker,
                        images: &images
                    )
                    if !text.isEmpty {
                        messages.append(("system", text))
                    }

                case .prompt(let prompt):
                    let text = try extractContent(
                        from: prompt.segments,
                        imageMarker: imageMarker,
                        images: &images
                    )
                    if !text.isEmpty {
                        messages.append(("user", text))
                    }

                case .reasoning:
                    // Keep display history in the transcript without sending unsupported replay state.
                    return
                case .response(let response):
                    let text = try extractContent(
                        from: response.segments,
                        imageMarker: imageMarker,
                        images: &images
                    )
                    if !text.isEmpty {
                        messages.append(("assistant", text))
                    }

                case .toolCalls(let toolCalls):
                    guard let toolContext else { break }
                    let parsed = toolCalls.map {
                        LlamaParsedToolCall(name: $0.toolName, argumentsJSON: $0.arguments.jsonString)
                    }
                    if let last = messages.last, last.role == "assistant" {
                        let markup = toolContext.format.assistantText(for: parsed, precededByContent: true)
                        messages[messages.count - 1].content += markup
                    } else {
                        let markup = toolContext.format.assistantText(for: parsed, precededByContent: false)
                        messages.append(("assistant", markup))
                    }

                case .toolOutput(let output):
                    guard let toolContext else { break }
                    let message = toolContext.format.toolResponseMessage(
                        toolName: output.toolName,
                        segments: output.segments
                    )
                    if let last = messages.last, last.role == message.role, last.role == "user",
                        last.content.hasSuffix("</tool_response>")
                    {
                        messages[messages.count - 1].content += "\n" + message.content
                    } else {
                        messages.append(message)
                    }
                }
            }

            for entry in requestContext.transcript {
                try appendEntry(entry)
            }
            if let toolContext {
                for entry in toolContext.pendingEntries {
                    try appendEntry(entry)
                }
            }

            if let toolContext, !toolContext.definitions.isEmpty {
                if let systemIndex = messages.firstIndex(where: { $0.role == "system" }) {
                    messages[systemIndex].content = try toolContext.format.systemMessage(
                        existingText: messages[systemIndex].content,
                        tools: toolContext.definitions
                    )
                } else {
                    let systemText = try toolContext.format.systemMessage(
                        existingText: "",
                        tools: toolContext.definitions
                    )
                    messages.insert(("system", systemText), at: 0)
                }
            }

            if let extraSystemMessage, !extraSystemMessage.isEmpty {
                messages.append(("system", extraSystemMessage))
            }

            // Keep C strings alive while using them
            let cRoles = messages.map { strdup($0.role) }
            let cContents = messages.map { strdup($0.content) }

            defer {
                cRoles.forEach { free($0) }
                cContents.forEach { free($0) }
            }

            var cMessages = [llama_chat_message]()
            for i in 0 ..< messages.count {
                cMessages.append(llama_chat_message(role: cRoles[i], content: cContents[i]))
            }

            // Get chat template embedded in the model's GGUF file (e.g., Llama 3, Mistral, ChatML)
            let tmpl = llama_model_chat_template(model, nil)

            // Get required buffer size
            let requiredSize = llama_chat_apply_template(
                tmpl,
                cMessages,
                cMessages.count,
                true,  // add_ass: Add assistant generation prompt
                nil,
                0
            )

            guard requiredSize > 0 else {
                if let tmpl, String(cString: tmpl).contains("<|turn>") {
                    return Self.renderGemma4Prompt(messages: messages, assistantPrefill: assistantPrefill)
                }
                throw LlamaLanguageModelError.encodingFailed
            }

            // Allocate buffer and apply template
            var buffer = [CChar](repeating: 0, count: Int(requiredSize) + 1)

            let result = llama_chat_apply_template(
                tmpl,
                cMessages,
                cMessages.count,
                true,
                &buffer,
                Int32(buffer.count)
            )

            guard result > 0 else {
                throw LlamaLanguageModelError.encodingFailed
            }

            let rendered = buffer.withUnsafeBytes { rawBuffer in
                String(decoding: rawBuffer.prefix(Int(result)), as: UTF8.self)
            }

            if let assistantPrefill, !assistantPrefill.isEmpty {
                return rendered + assistantPrefill
            }
            return rendered
        }

        /// Renders the Gemma 4 canonical chat format, which
        /// `llama_chat_apply_template` does not recognize: turns open with
        /// `<|turn>role`, close with `<turn|>`, and the assistant role is named
        /// `model`. The BOS token is applied during tokenization.
        static func renderGemma4Prompt(
            messages: [(role: String, content: String)],
            assistantPrefill: String?
        ) -> String {
            var rendered = ""
            var openModelTurn = false
            for (index, message) in messages.enumerated() {
                if message.role == "tool" {
                    rendered += message.content
                    continue
                }
                let role = message.role == "assistant" ? "model" : message.role
                let content = message.content.trimmingCharacters(in: .whitespacesAndNewlines)
                if role == "model" && openModelTurn {
                    rendered += content
                } else {
                    if openModelTurn {
                        rendered += "<turn|>\n"
                        openModelTurn = false
                    }
                    rendered += "<|turn>\(role)\n\(content)"
                }
                if role == "model" {
                    let nextRole = index + 1 < messages.count ? messages[index + 1].role : nil
                    if nextRole == "tool" || nextRole == "assistant" {
                        openModelTurn = true
                    } else {
                        rendered += "<turn|>\n"
                        openModelTurn = false
                    }
                } else {
                    rendered += "<turn|>\n"
                }
            }
            if !openModelTurn {
                rendered += "<|turn>model\n"
            }
            if let assistantPrefill, !assistantPrefill.isEmpty {
                rendered += assistantPrefill
            }
            return rendered
        }

        private func extractText(from segments: [Transcript.Segment]) -> String {
            segments.compactMap { segment -> String? in
                if case .text(let t) = segment { return t.content }
                return nil
            }.joined()
        }

        /// Extracts message content from segments, replacing each image segment
        /// with `imageMarker` and collecting its payload in order. Image segments
        /// throw ``LlamaLanguageModelError/unsupportedFeature`` when no marker is
        /// provided.
        private func extractContent(
            from segments: [Transcript.Segment],
            imageMarker: String?,
            images: inout [Data]
        ) throws -> String {
            var parts: [String] = []
            for segment in segments {
                switch segment {
                case .text(let t):
                    parts.append(t.content)
                case .image(let image):
                    guard let imageMarker else {
                        throw LlamaLanguageModelError.unsupportedFeature
                    }
                    switch image.source {
                    case .data(let data, _):
                        images.append(data)
                        parts.append(imageMarker)
                    case .url(let url):
                        guard url.isFileURL, let data = try? Data(contentsOf: url) else {
                            throw LlamaLanguageModelError.unsupportedFeature
                        }
                        images.append(data)
                        parts.append(imageMarker)
                    }
                default:
                    break
                }
            }
            return parts.joined()
        }

        private func tokenizeText(vocab: OpaquePointer, text: String) throws -> [llama_token] {
            let utf8Count = text.utf8.count
            let maxTokens = Int32(max(utf8Count * 2, 8))  // Rough estimate, minimum capacity
            let tokens = UnsafeMutablePointer<llama_token>.allocate(capacity: Int(maxTokens))
            defer { tokens.deallocate() }

            let tokenCount = llama_tokenize(
                vocab,
                text,
                Int32(utf8Count),
                tokens,
                maxTokens,
                true,  // addSpecial
                true  // parseSpecial
            )

            guard tokenCount > 0 else {
                throw LlamaLanguageModelError.tokenizationFailed
            }

            return Array(UnsafeBufferPointer(start: tokens, count: Int(tokenCount)))
        }

        /// 生成过程中的终止 token 文本（模板 EOG / 会话结构标签）。
        /// 部分 GGUF 未声明 `eos_token_id`，`llama_vocab_is_eog` 识别不到
        /// `<|im_end|>` 之类的模板 EOG，导致其被解码成 ◆ 混入正文。
        /// 此处按文本兜底识别：命中即结束生成且不 yield。
        private static let terminalTokenTexts: Set<String> = [
            "<|im_end|>", "<|im_start|>", "<|endoftext|>", "<|eot_id|>",
            "<|end|>", "<end>", "</s>", "<|tool_call|>", "<|tool_response|>",
        ]

        private func isTerminalTokenText(_ text: String) -> Bool {
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            return Self.terminalTokenTexts.contains(trimmed)
        }

        /// 输出 token 上限：用户显式设置时取较小者；未设置时按模型
        /// 原生上下文自动推导（预留 prompt 余量），不再写死固定值。
        private static func resolveMaxTokens(requested: Int?, contextSize: UInt32) -> Int {
            let cap = max(256, Int(contextSize) - 512)
            guard let requested else { return cap }
            return min(requested, cap)
        }

        private func tokenToText(vocab: OpaquePointer, token: llama_token) -> String? {
            // First attempt with a reasonable buffer
            var cap: Int32 = 64
            var buf = UnsafeMutablePointer<CChar>.allocate(capacity: Int(cap))
            defer { buf.deallocate() }

            var written = llama_token_to_piece(
                vocab,
                token,
                buf,
                cap,
                0,
                true
            )

            if written < 0 {
                // Reallocate to the required size and retry
                cap = -written
                buf.deallocate()
                buf = UnsafeMutablePointer<CChar>.allocate(capacity: Int(cap))
                written = llama_token_to_piece(
                    vocab,
                    token,
                    buf,
                    cap,
                    0,
                    true
                )
            }

            let count = Int(max(0, written))
            if count == 0 { return nil }

            // Create String from exact byte count (no reliance on NUL termination)
            let rawPtr = UnsafeRawPointer(buf)
            let u8Ptr = rawPtr.assumingMemoryBound(to: UInt8.self)
            let bytes = UnsafeBufferPointer(start: u8Ptr, count: count)
            return String(decoding: bytes, as: UTF8.self)
        }
    }

    /// Errors that can occur when using LlamaLanguageModel
    ///
    /// - Note: This API is exclusive to AnyLanguageModel
    ///   and using it means your code is no longer drop-in compatible
    ///   with the Foundation Models framework.
    public enum LlamaLanguageModelError: Error, LocalizedError {
        case modelLoadFailed
        case contextInitializationFailed
        case tokenizationFailed
        case encodingFailed
        case decodingFailed
        case invalidModelPath
        case insufficientMemory(requiredBytes: UInt64, availableBytes: UInt64)
        case promptExceedsContextWindow
        case unsupportedFeature
        case encoderOnlyModel

        public var errorDescription: String? {
            switch self {
            case .modelLoadFailed:
                return "Failed to load model from file"
            case .contextInitializationFailed:
                return "Failed to initialize context"
            case .tokenizationFailed:
                return "Failed to tokenize input text"
            case .encodingFailed:
                return "Failed to encode prompt"
            case .decodingFailed:
                return "Failed to decode response"
            case .invalidModelPath:
                return "Invalid model file path"
            case let .insufficientMemory(requiredBytes, availableBytes):
                if requiredBytes > 0 {
                    let gb = 1_073_741_824.0
                    let req = String(format: "%.1f", Double(requiredBytes) / gb)
                    let avail = String(format: "%.1f", Double(availableBytes) / gb)
                    return "Insufficient memory to load model (needs ~\(req) GB, \(avail) GB available). Close other apps or use a smaller model."
                }
                return "Insufficient memory for operation"
            case .promptExceedsContextWindow:
                return "Prompt is longer than the model's context window"
            case .unsupportedFeature:
                return "This LlamaLanguageModel does not support image segments"
            case .encoderOnlyModel:
                return "This model is encoder-only (e.g., BERT) and cannot generate text"
            }
        }
    }
#endif  // Llama
