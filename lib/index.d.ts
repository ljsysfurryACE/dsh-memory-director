/**
 * dsh-memory-director — MemoryDirector plugin for DeepSeek Harness.
 *
 * Official Harness compacts history via summarization, but has NO concept of
 * "which facts are worth remembering across sessions". This plugin adds the
 * AgentFrame MemoryDirector: after each turn, an LLM decides what to
 * remember / forget; before each step, relevant memories are injected into
 * the model context.
 *
 * @module @agentframe/dsh-memory-director
 */
import { Context } from '@deepseek-ai/cordis';
import z from '@deepseek-ai/schemastery';
/** One persisted memory entry. */
export interface MemoryEntry {
    id: string;
    text: string;
    importance: number;
    accessCount: number;
    createdAt: number;
    lastAccessAt: number;
}
export interface MemoryDirectorConfig {
    /** Provider to use for memory decisions. */
    provider: string;
    /** Model to use for memory decisions. */
    model: string;
    /** Max tokens for the decision call. */
    maxTokens: number;
    /** Cosine-similarity dedup threshold. */
    dedupThreshold: number;
    /** Forget threshold on importance decay. */
    forgetThreshold: number;
    /** Memory store file path. */
    storePath: string;
    /** Enable turn-end auto decisions. */
    auto: boolean;
}
/**
 * MemoryDirectorService — exposes ctx.memory (remember/search/forget)
 * and hooks the agent loop to auto-manage memory.
 */
export declare class MemoryDirectorService {
    private readonly ctx;
    static inject: string[];
    static Config: z<MemoryDirectorConfig>;
    readonly config: MemoryDirectorConfig;
    private memories;
    private readonly llm;
    constructor(ctx: Context, config?: Partial<MemoryDirectorConfig>);
    remember(text: string, importance?: number): MemoryEntry;
    forget(id: string): boolean;
    search(query: string, limit?: number): MemoryEntry[];
    all(): MemoryEntry[];
    clear(): void;
    decide(turnText: string): Promise<{
        remember: string[];
        forget: string[];
        importance: number;
    }>;
    private _hookAgentLoop;
    private _turnText;
    private _eventText;
    private _tokenize;
    private _findSimilar;
    private _load;
    private _save;
}
export default MemoryDirectorService;
