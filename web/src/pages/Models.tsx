import { useState } from "react";
import { useNavigate } from "react-router";

import { ModelBrowser } from "../components/ModelBrowser";
import { ModelDetail } from "../components/ModelDetail";
import { Button, Drawer, SectionTitle, useToast } from "../components/ui";
import { api } from "../lib/api";
import { MODALITY_LABEL } from "../lib/format";
import type { CatalogItem, Modality } from "../lib/types";

export default function Models() {
  const [modality, setModality] = useState<Modality>("text");
  const [open, setOpen] = useState<CatalogItem | null>(null);
  const [busy, setBusy] = useState(false);
  const nav = useNavigate();
  const toast = useToast();

  return (
    <div>
      <SectionTitle kicker="Catalog" title="Models">
        <p className="max-w-md text-[13.5px] text-fg-2">
          Live from Hugging Face and Ollama, plus anything attached under <code className="font-mono text-[12px]">/kaggle/input</code>. Each model is sized against this machine.
        </p>
      </SectionTitle>
      <ModelBrowser modality={modality} onModality={setModality} onOpen={setOpen} />
      <Drawer open={!!open} onClose={() => setOpen(null)} title={open ? `${MODALITY_LABEL[open.modality]} model` : ""}>
        {open && (
          <ModelDetail
            key={open.key}
            item={open}
            busy={busy}
            actionLabel="Deploy now"
            onAction={async (c) => {
              setBusy(true);
              try {
                const dep = await api.deploy({ ...c.spec, interface: "studio" });
                toast(`Launching ${dep.name}…`);
                nav(`/deployments#${dep.id}`);
              } catch (e) {
                toast(e instanceof Error ? e.message : "Deploy failed", "error");
              } finally {
                setBusy(false);
              }
            }}
            secondary={(c) => (
              <Button variant="ghost" onClick={() => nav("/", { state: { choice: c } })}>
                Customize…
              </Button>
            )}
          />
        )}
      </Drawer>
    </div>
  );
}
