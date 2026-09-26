import { useQueryClient } from "@tanstack/react-query";
import { ArrowRight, Pencil, Plus, Trash2 } from "lucide-react";
import { useState, type ReactNode } from "react";
import { Card, InlineError, PageBody, PageTitle, Row, Rows, Section } from "../components/parts";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";
import { Select } from "../components/ui/select";
import { Switch } from "../components/ui/switch";
import { api, errorText } from "../lib/api";
import { isDirty } from "../lib/draft";
import { useStore, useSyncedDraft } from "../lib/store";
import type { Dictionary, DictionaryRule, LiteralRule } from "../lib/types";

const OPTIONS: { key: keyof Dictionary["options"]; title: string; description: string }[] = [
  { key: "remove_fillers", title: "Remove fillers", description: "Drops um, uh, uhh, umm and erm." },
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

/** Mirrors the old page: "a, b" -> ["a", "b"]; a space at either end of the box glues every alias to the neighbouring text. */
export function splitAliases(spoken: string): string[] {
  const [left, right] = edgeWhitespace(spoken);
  return spoken.split(",").map((alias) => alias.trim()).filter(Boolean).map((alias) => left + alias + right);
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
    learned: original?.kind === "literal" && original.learned && sameAliases(original.spoken, spoken) && original.replacement === editor.replacement,
  };
  return rule;
}

function sameAliases(a: string[], b: string[]): boolean {
  return a.length === b.length && a.every((alias, index) => alias === b[index]);
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

export function DictionaryPage() {
  const { dictionary, dispatchDictionary, saveDictionary, confirm } = useStore();
  const client = useQueryClient();
  const failed = useSyncedDraft<Dictionary>("dictionary", "/api/dictionary", dispatchDictionary);
  const [adding, setAdding] = useState<Editor>(EMPTY_EDITOR);
  const [editing, setEditing] = useState<{ index: number; editor: Editor } | null>(null);
  const [error, setError] = useState("");
  const draft = dictionary?.draft;
  if (!draft) return <PageBody><PageTitle title="Dictionary" />{failed ? <InlineError>The dictionary could not be loaded.</InlineError> : null}</PageBody>;

  const editRules = (update: (rules: DictionaryRule[]) => DictionaryRule[]) =>
    dispatchDictionary({ type: "edit", update: (current) => ({ ...current, rules: update(current.rules) }) });
  const place = (editor: Editor, index: number): boolean => {
    const rule = ruleFrom(editor, draft.rules[index]);
    if (typeof rule === "string") return setError(rule), false;
    if (duplicateAlias(draft.rules, rule, index)) return setError("Another word rule already uses one of those aliases."), false;
    setError("");
    editRules((rules) => (index < rules.length ? rules.map((other, at) => (at === index ? rule : other)) : [...rules, rule]));
    return true;
  };
  const forgetLearned = async () => {
    if (!(await confirm("Forget all learned words?", "Removes every word the app learned from your corrections. It will not learn them again.", "Forget"))) return;
    try {
      await api.post("/api/dictionary/forget-learned");
      const fresh = await api.get<Dictionary>("/api/dictionary");
      editRules((rules) => rules.filter((rule) => !(rule.kind === "literal" && rule.learned)));
      dispatchDictionary({ type: "loaded", value: fresh });
      client.setQueryData(["dictionary"], fresh);
    } catch (failure) {
      setError(errorText(failure));
    }
  };

  return (
    <PageBody>
      <PageTitle title="Dictionary" />
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
        title="Words and replacements"
        note="Replace spoken words or phrases with typed text. Separate aliases with commas; regex rules use one pattern. A space at the start or end of the box glues every alias to the neighbouring text."
      >
        <Card className="divide-y">
          <form
            className="px-4 py-3"
            onSubmit={(event) => {
              event.preventDefault();
              if (place(adding, draft.rules.length)) setAdding({ ...EMPTY_EDITOR, kind: adding.kind });
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
            draft.rules.map((rule, index) =>
              editing?.index === index ? (
                <div key={index} className="bg-muted/40 px-4 py-3">
                  <RuleEditor editor={editing.editor} onChange={(editor) => setEditing({ index, editor })}>
                    <Button size="sm" variant="ghost" onClick={() => (setEditing(null), setError(""))}>
                      Cancel
                    </Button>
                    <Button size="sm" onClick={() => place(editing.editor, index) && setEditing(null)}>
                      Done
                    </Button>
                  </RuleEditor>
                </div>
              ) : (
                <div key={index} className="grid grid-cols-[minmax(0,1fr)_16px_minmax(0,1fr)_172px] items-center gap-2 px-4 py-2">
                  <div className="flex min-w-0 flex-wrap gap-1">
                    {rule.kind === "literal" ? (
                      rule.spoken.map((alias, at) => (
                        <span key={`${alias}-${at}`} className="max-w-full truncate rounded-md bg-muted px-1.5 py-px text-[12.5px]">
                          {shown(alias)}
                        </span>
                      ))
                    ) : (
                      <code className="truncate rounded-md bg-muted px-1.5 py-px font-mono text-[12px]">{shown(rule.pattern)}</code>
                    )}
                  </div>
                  <ArrowRight className="size-3.5 text-muted-foreground" />
                  <span className="truncate text-[13px]">{rule.replacement || <span className="text-muted-foreground">(nothing)</span>}</span>
                  <span className="flex items-center justify-end gap-1">
                    {rule.kind === "regex" ? <Badge>Regex</Badge> : rule.learned ? <Badge variant="primary">Learned</Badge> : null}
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
            <p className="px-4 py-6 text-center text-[12.5px] text-muted-foreground">No words yet. Add one above, or correct a pasted word and it is learned.</p>
          )}
        </Card>
        {error ? <InlineError>{error}</InlineError> : null}
        <div className="mt-2.5 flex items-center gap-2">
          <Button variant="link" onClick={() => void forgetLearned()}>
            Forget all learned words
          </Button>
          {isDirty(dictionary) ? (
            <>
              <span className="ml-auto text-[12.5px] text-muted-foreground">Dictionary changed</span>
              <Button variant="ghost" size="sm" onClick={() => (dispatchDictionary({ type: "discard" }), setError(""))}>
                Discard
              </Button>
              <Button size="sm" onClick={async () => setError(await saveDictionary())}>
                Save dictionary
              </Button>
            </>
          ) : null}
        </div>
      </Section>
    </PageBody>
  );
}
