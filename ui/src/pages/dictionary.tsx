import { useQueryClient } from "@tanstack/react-query";
import { ArrowRight, Check as CheckIcon, LoaderCircle, Pencil, Plus, Trash2, X } from "lucide-react";
import { useState, type ReactNode } from "react";
import { Card, InlineError, PageBody, PageTitle, Row, Rows, Section } from "../components/parts";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";
import { Select } from "../components/ui/select";
import { Switch } from "../components/ui/switch";
import { api, errorText } from "../lib/api";
import { useStore, useSyncedDraft, type DictionarySave } from "../lib/store";
import type { Dictionary, DictionaryRule, LiteralRule, Suggestion, Term } from "../lib/types";

const OPTIONS: { key: keyof Dictionary["options"]; title: string; description: string }[] = [
  { key: "remove_fillers", title: "Remove fillers", description: "Drops um, uh, ah and erm, and short words said twice by accident, like “the the”." },
  { key: "spoken_commands", title: "Spoken commands", description: "Say “new line” or “period” to type them." },
  { key: "numbers", title: "Format numbers", description: "Writes “twenty five” as 25." },
  { key: "punctuation_model", title: "Restore punctuation", description: "Adds punctuation with a small extra model." },
];

const KINDS = [
  { value: "literal", label: "Words" },
  { value: "regex", label: "Regex" },
];

function edgeWhitespace(text: string): [string, string] {
  return [text.slice(0, text.length - text.trimStart().length), text.slice(text.trimEnd().length)];
}

/** Mirrors the old page: "a, b" -> ["a", "b"]; a space at either end of the box glues every alias to the neighbouring text.
 * A comma at the very start or end of the box belongs to the alias, so ", underscore" also swallows the comma before it. */
export function splitAliases(spoken: string): string[] {
  const [left, right] = edgeWhitespace(spoken);
  const text = spoken.trim();
  const head = /^(?:,\s*)*/.exec(text)?.[0] ?? "";
  const tail = /(?:\s*,)*$/.exec(text.slice(head.length))?.[0] ?? "";
  const aliases = text.slice(head.length, text.length - tail.length).split(",").map((alias) => alias.trim()).filter(Boolean);
  const last = aliases.length - 1;
  return aliases.map((alias, at) => left + (at === 0 ? head : "") + alias + (at === last ? tail : "") + right);
}

export function joinAliases(aliases: string[]): string {
  const [left] = edgeWhitespace(aliases[0] ?? "");
  const [, right] = edgeWhitespace(aliases.at(-1) ?? "");
  return left + aliases.map((alias) => alias.trim()).join(", ") + right;
}

function shown(source: string): string {
  return source === source.trim() ? source : `"${source}"`;
}

export function duplicateAlias(rules: DictionaryRule[], candidate: DictionaryRule, ignoredIndex: number): boolean {
  if (candidate.kind !== "literal") return false;
  const aliases = new Set(candidate.spoken.map((alias) => alias.trim()));
  return rules.some((rule, index) => index !== ignoredIndex && rule.kind === "literal" && rule.spoken.some((alias) => aliases.has(alias.trim())));
}

interface Editor {
  kind: "literal" | "regex";
  spoken: string;
  replacement: string;
  case_sensitive: boolean;
  whole_words: boolean;
  phonetic: boolean;
}

const EMPTY_EDITOR: Editor = { kind: "literal", spoken: "", replacement: "", case_sensitive: false, whole_words: true, phonetic: false };

function editorFor(rule: DictionaryRule): Editor {
  if (rule.kind === "regex") return { ...EMPTY_EDITOR, kind: "regex", spoken: rule.pattern, replacement: rule.replacement, whole_words: false };
  return { kind: "literal", spoken: joinAliases(rule.spoken), replacement: rule.replacement, case_sensitive: rule.case_sensitive, whole_words: rule.whole_words, phonetic: rule.phonetic };
}

function ruleFrom(editor: Editor, original: DictionaryRule | undefined): DictionaryRule | string {
  if (!editor.spoken.trim()) return "At least one spoken alias or regex pattern is required.";
  if (editor.kind === "regex") return { kind: "regex", pattern: editor.spoken, replacement: editor.replacement, enabled: original?.enabled ?? true };
  const spoken = splitAliases(editor.spoken);
  if (!spoken.length) return "At least one spoken alias or regex pattern is required.";
  const rule: LiteralRule = {
    kind: "literal",
    spoken,
    replacement: editor.replacement,
    case_sensitive: editor.case_sensitive,
    whole_words: editor.whole_words,
    phonetic: editor.phonetic,
    enabled: original?.enabled ?? true,
  };
  return rule;
}

interface TermEditor {
  written: string;
  note: string;
  heard: string;
  phonetic: boolean;
}

const EMPTY_TERM: TermEditor = { written: "", note: "", heard: "", phonetic: false };

function termEditorFor(term: Term): TermEditor {
  return { written: term.written, note: term.note, heard: term.heard_as.join(", "), phonetic: term.phonetic };
}

/** The entry as typed: trimmed, its heard-as spellings split on commas, without repeats or its own spelling. */
export function termFrom(editor: TermEditor): Term {
  const written = editor.written.trim();
  const seen = new Set([written.toLowerCase()]);
  const heard_as = editor.heard.split(",").map((spelling) => spelling.trim()).filter((spelling) => {
    const key = spelling.toLowerCase();
    if (!spelling || seen.has(key)) return false;
    seen.add(key);
    return true;
  });
  return { written, note: editor.note.trim().replace(/\s+/g, " "), heard_as, phonetic: editor.phonetic };
}

/** Why *candidate* can't join *terms* (ignoring the one at *ignoredIndex*), or "" when it can. The core checks the same on save. */
export function termProblem(terms: Term[], candidate: Term, ignoredIndex: number): string {
  if (!candidate.written) return "Write the name or term the way it should appear.";
  const others = terms.filter((_, index) => index !== ignoredIndex);
  const key = (text: string) => text.toLowerCase();
  if (others.some((term) => key(term.written) === key(candidate.written))) return `“${candidate.written}” is already in the list.`;
  for (const spelling of candidate.heard_as) {
    if (others.some((term) => key(term.written) === key(spelling))) return `“${spelling}” is an entry of its own, so it can't be swapped for “${candidate.written}”.`;
    const owner = others.find((term) => term.heard_as.some((other) => key(other) === key(spelling)));
    if (owner) return `“${spelling}” is already heard as “${owner.written}”.`;
  }
  const taken = others.find((term) => term.heard_as.some((spelling) => key(spelling) === key(candidate.written)));
  return taken ? `“${candidate.written}” is a heard-as spelling of “${taken.written}”. Remove it there first.` : "";
}

function Check({ label, checked, disabled, onChange }: { label: string; checked: boolean; disabled?: boolean; onChange: (checked: boolean) => void }) {
  return (
    <label className="flex items-center gap-1.5 text-[12.5px] has-disabled:opacity-50">
      <input type="checkbox" className="size-3.5 accent-primary" checked={checked} disabled={disabled} onChange={(event) => onChange(event.target.checked)} />
      {label}
    </label>
  );
}

function RuleEditor({ editor, onChange, children }: { editor: Editor; onChange: (editor: Editor) => void; children: ReactNode }) {
  const literal = editor.kind === "literal";
  return (
    <div className="flex flex-col gap-2">
      <div className="grid grid-cols-[minmax(0,1fr)_16px_minmax(0,1fr)_172px] items-center gap-2">
        <Input
          aria-label="When I say"
          placeholder={literal ? "When I say, e.g. jason, jayson" : "Pattern, e.g. \\bteh\\b"}
          value={editor.spoken}
          onChange={(event) => onChange({ ...editor, spoken: event.target.value })}
        />
        <ArrowRight className="size-3.5 text-muted-foreground" />
        <Input aria-label="Replace with" placeholder="Replace with" value={editor.replacement} onChange={(event) => onChange({ ...editor, replacement: event.target.value })} />
        <Select label="Rule type" value={editor.kind} options={KINDS} onChange={(kind) => onChange({ ...editor, kind: kind as Editor["kind"] })} />
      </div>
      <div className="flex items-center gap-4">
        <Check label="Case sensitive" checked={literal && editor.case_sensitive} disabled={!literal} onChange={(case_sensitive) => onChange({ ...editor, case_sensitive })} />
        <Check label="Whole words" checked={literal && editor.whole_words} disabled={!literal} onChange={(whole_words) => onChange({ ...editor, whole_words })} />
        <Check label="Sounds alike" checked={literal && editor.phonetic} disabled={!literal} onChange={(phonetic) => onChange({ ...editor, phonetic })} />
        <span className="ml-auto flex gap-2">{children}</span>
      </div>
    </div>
  );
}

function TermFields({ editor, onChange, children }: { editor: TermEditor; onChange: (editor: TermEditor) => void; children: ReactNode }) {
  return (
    <div className="flex flex-col gap-2">
      <div className="grid grid-cols-[minmax(0,1fr)_minmax(0,1fr)_minmax(0,1fr)] items-center gap-2">
        <Input aria-label="Written as" placeholder="Written as, e.g. Skryb" value={editor.written} onChange={(event) => onChange({ ...editor, written: event.target.value })} />
        <Input aria-label="Note" placeholder="Note, e.g. the app I'm building" value={editor.note} onChange={(event) => onChange({ ...editor, note: event.target.value })} />
        <Input aria-label="Heard as" placeholder="Heard as, e.g. skribe, scryb" value={editor.heard} onChange={(event) => onChange({ ...editor, heard: event.target.value })} />
      </div>
      <div className="flex items-center gap-4">
        <Check label="Sounds alike" checked={editor.phonetic} onChange={(phonetic) => onChange({ ...editor, phonetic })} />
        <span className="ml-auto flex gap-2">{children}</span>
      </div>
    </div>
  );
}

function SuggestionRow({ suggestion, onAccept, onDismiss }: { suggestion: Suggestion; onAccept: () => void; onDismiss: () => void }) {
  return (
    <div className="flex items-center gap-2 bg-primary/5 px-4 py-2">
      <span className="min-w-0 flex-1 truncate text-[13px]">
        Swap “{suggestion.heard}” for <span className="font-medium">{suggestion.written}</span>
      </span>
      <span className="text-[12px] text-muted-foreground">{suggestion.times === 1 ? "re-dictated once" : `re-dictated ${suggestion.times} times`}</span>
      <Button size="sm" onClick={onAccept}>
        <CheckIcon />
        Accept
      </Button>
      <Button variant="ghost" size="icon" aria-label="Dismiss suggestion" title="Dismiss suggestion" onClick={onDismiss}>
        <X />
      </Button>
    </div>
  );
}

/** A quiet word on the dictionary's own saving, and what went wrong when a save didn't work. */
function SaveCue({ save, onRetry }: { save: DictionarySave; onRetry: () => void }) {
  if (save.state === "saving") {
    return (
      <span className="flex items-center gap-1.5 text-[12.5px] text-muted-foreground">
        <LoaderCircle className="size-3.5 animate-spin" />
        Saving...
      </span>
    );
  }
  if (save.state === "saved") {
    return (
      <span className="flex items-center gap-1.5 text-[12.5px] text-muted-foreground">
        <CheckIcon className="size-3.5" />
        Saved
      </span>
    );
  }
  if (save.state === "failed") {
    return (
      <span role="alert" className="flex min-w-0 items-center gap-2 text-[12.5px] text-destructive">
        <span className="min-w-0">{save.error}</span>
        {save.retry ? (
          <Button size="sm" variant="outline" onClick={onRetry}>
            Retry
          </Button>
        ) : null}
      </span>
    );
  }
  return null;
}

/** The list shows the newest rule first; the saved order, which rules apply in, stays oldest first. */
export function newestFirst<T>(rules: T[]): { rule: T; index: number }[] {
  return rules.map((rule, index) => ({ rule, index })).reverse();
}

export function DictionaryPage() {
  const { dictionary, dispatchDictionary, flushDictionary, dictionarySave } = useStore();
  const client = useQueryClient();
  const failed = useSyncedDraft<Dictionary>("dictionary", "/api/dictionary", dispatchDictionary);
  const [adding, setAdding] = useState<Editor>(EMPTY_EDITOR);
  const [editing, setEditing] = useState<{ index: number; editor: Editor } | null>(null);
  const [addingTerm, setAddingTerm] = useState<TermEditor>(EMPTY_TERM);
  const [editingTerm, setEditingTerm] = useState<{ index: number; editor: TermEditor } | null>(null);
  const [error, setError] = useState("");
  const [termError, setTermError] = useState("");
  const draft = dictionary?.draft;
  if (!draft) return <PageBody><PageTitle title="Dictionary" />{failed ? <InlineError>The dictionary could not be loaded.</InlineError> : null}</PageBody>;

  const editRules = (update: (rules: DictionaryRule[]) => DictionaryRule[]) =>
    dispatchDictionary({ type: "edit", update: (current) => ({ ...current, rules: update(current.rules) }) });
  /**
   * Adding or changing an entry saves it at once, after anything still waiting, so one the core refuses (a regex
   * it can't read) takes nothing else back with it and stays in its box to fix.
   */
  const saveNow = async (change: () => void): Promise<string> => {
    await flushDictionary();
    change();
    return flushDictionary();
  };
  const place = async (editor: Editor, index: number): Promise<boolean> => {
    const rule = ruleFrom(editor, draft.rules[index]);
    if (typeof rule === "string") return setError(rule), false;
    if (duplicateAlias(draft.rules, rule, index)) return setError("Another word rule already uses one of those aliases."), false;
    const problem = await saveNow(() =>
      editRules((rules) => (index < rules.length ? rules.map((other, at) => (at === index ? rule : other)) : [...rules, rule])),
    );
    setError(problem);
    return !problem;
  };
  const editTerms = (update: (people: Term[]) => Term[]) =>
    dispatchDictionary({ type: "edit", update: (current) => ({ ...current, people: update(current.people) }) });
  const placeTerm = async (editor: TermEditor, index: number): Promise<boolean> => {
    const term = termFrom(editor);
    const invalid = termProblem(draft.people, term, index);
    if (invalid) return setTermError(invalid), false;
    const problem = await saveNow(() => editTerms((people) => (index < people.length ? people.map((other, at) => (at === index ? term : other)) : [...people, term])));
    setTermError(problem);
    return !problem;
  };
  /** Settled by the core itself, after any change still on its way there, so neither undoes the other. */
  const settle = async (request: () => Promise<void>, change: (current: Dictionary) => Dictionary) => {
    try {
      await flushDictionary();
      await request();
      const fresh = await api.get<Dictionary>("/api/dictionary");
      dispatchDictionary({ type: "edit", update: (current) => ({ ...change(current), suggestions: fresh.suggestions }) });
      dispatchDictionary({ type: "loaded", value: fresh });
      client.setQueryData(["dictionary"], fresh);
    } catch (failure) {
      setTermError(errorText(failure));
    }
  };
  const path = (suggestion: Suggestion) => `/api/people/suggestions/${encodeURIComponent(suggestion.heard)}`;
  const accept = (suggestion: Suggestion) =>
    settle(() => api.post(`${path(suggestion)}/accept`), (current) => ({
      ...current,
      people: current.people.map((term) => (term.written === suggestion.written ? { ...term, heard_as: [...term.heard_as, suggestion.heard] } : term)),
    }));
  const dismiss = (suggestion: Suggestion) => settle(() => api.del(path(suggestion)), (current) => current);

  return (
    <PageBody>
      <PageTitle title="Dictionary">
        <SaveCue save={dictionarySave} onRetry={() => void flushDictionary()} />
      </PageTitle>
      <Section title="Cleanup">
        <Rows>
          {OPTIONS.map(({ key, title, description }) => (
            <Row key={key} title={title} description={description}>
              <Switch
                aria-label={title}
                checked={draft.options[key]}
                onCheckedChange={(on) => dispatchDictionary({ type: "edit", update: (current) => ({ ...current, options: { ...current.options, [key]: on } }) })}
              />
            </Row>
          ))}
        </Rows>
      </Section>

      <Section
        title="People and terms"
        note="Names and words dictation and meetings should spell right. Dictation and meetings tell the model about them, notes included, so it can tell your app from the ordinary word. Heard-as spellings are swapped for the entry. Dictating over a selected name again picks the other spelling, and suggests what it heard here."
      >
        <Card className="divide-y">
          {draft.suggestions.map((suggestion) => (
            <SuggestionRow key={suggestion.heard} suggestion={suggestion} onAccept={() => void accept(suggestion)} onDismiss={() => void dismiss(suggestion)} />
          ))}
          <form
            className="px-4 py-3"
            onSubmit={async (event) => {
              event.preventDefault();
              if (await placeTerm(addingTerm, draft.people.length)) setAddingTerm(EMPTY_TERM);
            }}
          >
            <TermFields editor={addingTerm} onChange={setAddingTerm}>
              <Button type="submit" size="sm" disabled={!addingTerm.written.trim()}>
                <Plus />
                Add
              </Button>
            </TermFields>
          </form>
          {draft.people.length ? (
            newestFirst(draft.people).map(({ rule: term, index }) =>
              editingTerm?.index === index ? (
                <div key={index} className="bg-muted/40 px-4 py-3">
                  <TermFields editor={editingTerm.editor} onChange={(editor) => setEditingTerm({ index, editor })}>
                    <Button size="sm" variant="ghost" onClick={() => (setEditingTerm(null), setTermError(""))}>
                      Cancel
                    </Button>
                    <Button size="sm" onClick={async () => (await placeTerm(editingTerm.editor, index)) && setEditingTerm(null)}>
                      Done
                    </Button>
                  </TermFields>
                </div>
              ) : (
                <div key={index} className="flex items-center gap-2 px-4 py-2">
                  <span className="truncate text-[13px] font-medium">{term.written}</span>
                  {term.note ? <span className="truncate text-[12.5px] text-muted-foreground">{term.note}</span> : null}
                  <span className="flex min-w-0 flex-1 flex-wrap justify-end gap-1">
                    {term.heard_as.map((spelling) => (
                      <span key={spelling} className="chip max-w-full truncate px-1.5 py-px text-[12.5px]">
                        {spelling}
                      </span>
                    ))}
                  </span>
                  {term.phonetic ? <Badge>Sounds alike</Badge> : null}
                  <Button variant="ghost" size="icon" aria-label="Edit entry" onClick={() => setEditingTerm({ index, editor: termEditorFor(term) })}>
                    <Pencil />
                  </Button>
                  <Button variant="ghost" size="icon" aria-label="Delete entry" onClick={() => editTerms((people) => people.filter((_, at) => at !== index))}>
                    <Trash2 />
                  </Button>
                </div>
              ),
            )
          ) : (
            <p className="px-4 py-6 text-center text-[12.5px] text-muted-foreground">No names yet. Add the people, products and terms you say most.</p>
          )}
        </Card>
        {termError ? <InlineError>{termError}</InlineError> : null}
      </Section>

      <Section
        title="Words and replacements"
        note="Replace spoken words or phrases with typed text. Separate aliases with commas; regex rules use one pattern. A space at the start or end of the box glues every alias to the neighbouring text."
      >
        <Card className="divide-y">
          <form
            className="px-4 py-3"
            onSubmit={async (event) => {
              event.preventDefault();
              if (await place(adding, draft.rules.length)) setAdding({ ...EMPTY_EDITOR, kind: adding.kind });
            }}
          >
            <RuleEditor editor={adding} onChange={setAdding}>
              <Button type="submit" size="sm" disabled={!adding.spoken.trim()}>
                <Plus />
                Add
              </Button>
            </RuleEditor>
          </form>
          {draft.rules.length ? (
            newestFirst(draft.rules).map(({ rule, index }) =>
              editing?.index === index ? (
                <div key={index} className="bg-muted/40 px-4 py-3">
                  <RuleEditor editor={editing.editor} onChange={(editor) => setEditing({ index, editor })}>
                    <Button size="sm" variant="ghost" onClick={() => (setEditing(null), setError(""))}>
                      Cancel
                    </Button>
                    <Button size="sm" onClick={async () => (await place(editing.editor, index)) && setEditing(null)}>
                      Done
                    </Button>
                  </RuleEditor>
                </div>
              ) : (
                <div key={index} className="grid grid-cols-[minmax(0,1fr)_16px_minmax(0,1fr)_172px] items-center gap-2 px-4 py-2">
                  <div className="flex min-w-0 flex-wrap gap-1">
                    {rule.kind === "literal" ? (
                      rule.spoken.map((alias, at) => (
                        <span key={`${alias}-${at}`} className="chip max-w-full truncate px-1.5 py-px text-[12.5px]">
                          {shown(alias)}
                        </span>
                      ))
                    ) : (
                      <code className="chip truncate px-1.5 py-px font-mono text-[12px]">{shown(rule.pattern)}</code>
                    )}
                  </div>
                  <ArrowRight className="size-3.5 text-muted-foreground" />
                  <span className="truncate text-[13px]">{rule.replacement || <span className="text-muted-foreground">(nothing)</span>}</span>
                  <span className="flex items-center justify-end gap-1">
                    <span className="text-[11.5px] text-muted-foreground" title="Times this rule has changed a dictation">
                      {rule.fired ? `${rule.fired}×` : "never"}
                    </span>
                    {rule.kind === "regex" ? <Badge>Regex</Badge> : null}
                    <Switch
                      aria-label="Rule on"
                      className="mx-1 scale-90"
                      checked={rule.enabled}
                      onCheckedChange={(enabled) => editRules((rules) => rules.map((other, at) => (at === index ? { ...other, enabled } : other)))}
                    />
                    <Button variant="ghost" size="icon" aria-label="Edit rule" onClick={() => setEditing({ index, editor: editorFor(rule) })}>
                      <Pencil />
                    </Button>
                    <Button variant="ghost" size="icon" aria-label="Delete rule" onClick={() => editRules((rules) => rules.filter((_, at) => at !== index))}>
                      <Trash2 />
                    </Button>
                  </span>
                </div>
              ),
            )
          ) : (
            <p className="px-4 py-6 text-center text-[12.5px] text-muted-foreground">No words yet. Add one above.</p>
          )}
        </Card>
        {error ? <InlineError>{error}</InlineError> : null}
      </Section>
    </PageBody>
  );
}
