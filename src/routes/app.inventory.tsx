import { useState, useEffect, useCallback, useMemo, useRef } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { toast } from "sonner";
import {
  Sparkles,
  Truck,
  AlertTriangle,
  CheckCircle2,
  Package,
  RefreshCw,
  UploadCloud,
  Trash2,
  Layers,
  Info,
  Calendar,
  Search,
  SlidersHorizontal,
  ChevronRight,
  Database,
  ArrowUpRight,
  ShieldCheck,
  ClipboardList,
  Plus,
  Minus,
  TrendingUp,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import {
  PageHeader,
  Panel,
  StatCard,
  StatusPill,
} from "@/components/app-ui";
import { auth } from "@/lib/auth";

export const Route = createFileRoute("/app/inventory")({
  head: () => ({
    meta: [
      { title: "Inventory Management — SmartBPI" },
      {
        name: "description",
        content:
          "Live stock monitoring, supplier lead times, purchase orders, and AI demand forecasting powered by raj_business_demand_model.joblib.",
      },
      {
        property: "og:title",
        content: "Inventory Management — SmartBPI",
      },
      {
        property: "og:description",
        content:
          "Keep every SKU above its reorder threshold with evidence-based AI demand prediction.",
      },
    ],
  }),
  component: InventoryManagement,
});

const API_BASE_URL =
  (import.meta.env["VITE_FORECAST_API_URL"] as string | undefined) ||
  "http://127.0.0.1:8000";

// ============================================================================
// Types
// ============================================================================

interface ModelInfo {
  model_file: string;
  model_type: string;
  trained_through: string;
  sklearn_version: string;
  forecast_horizon_days: number;
  feature_cols: string[];
  test_metrics: {
    training_r2?: number;
    testing_r2?: number;
    test_mae?: number;
    test_rmse?: number;
  };
}

interface ProductItem {
  sku: string;
  product_id: string;
  store_id: string;
  name: string;
  category: string;
  region: string;
  current_stock: number;
  reorder_threshold: number;
  unit_price: number;
  supplier: string;
  supplier_lead_time_days: number;
  units_ordered: number;
  daily_velocity: number;
  last_sales_date: string;
  latest_daily_sales: number;
}

interface StockMovement {
  id: string;
  sku: string;
  type: "In" | "Out";
  qty: number;
  ref: string;
  date: string;
}

interface SupplierItem {
  name: string;
  lead: string;
  reliability: string;
  terms: string;
  spend: string;
  active_skus: number;
}

interface PurchaseOrderItem {
  id: string;
  supplier: string;
  sku: string;
  product_name: string;
  qty: number;
  value: number;
  status: "In Transit" | "Confirmed" | "Draft";
}

interface CatalogSummary {
  total_skus: number;
  low_stock_count: number;
  total_valuation: number;
  avg_cover_days: number;
}

interface CatalogResponse {
  has_data: boolean;
  data_source?: string | null;
  dataset_available_to_load?: boolean;
  store_id?: string;
  latest_data_date?: string;
  summary: CatalogSummary;
  products: ProductItem[];
  stock_movements: StockMovement[];
  suppliers: SupplierItem[];
  purchase_orders: PurchaseOrderItem[];
  message?: string;
}

interface ReplenishmentAdvice {
  status: "sufficient" | "reorder_recommended" | "critical";
  recommended_reorder_qty: number;
  days_until_stockout: number;
  lead_time_demand: number;
  safety_target: number;
  reason: string;
}

interface ProductPredictionResult {
  product_id: string;
  sku: string;
  store_id: string;
  category: string;
  unit_price: number;
  target_date: string;
  historical_end_date: string;
  historical_records_used: number;
  predicted_demand: number;
  features_used: {
    product_id: string;
    store_id: string;
    lag_1: number;
    lag_7: number;
    lag_14: number;
    lag_28: number;
    rolling_mean_7: number;
    day_of_week: number;
    month: number;
    day_of_year: number;
  };
  current_stock: number;
  reorder_threshold: number;
  supplier_lead_time_days: number;
  pending_orders: number;
  replenishment: ReplenishmentAdvice;
  model_type: string;
  trained_through: string;
}

// Currency Formatter (INR for Indian business context)
function formatCurrency(value: number | null | undefined): string {
  if (value === null || value === undefined || isNaN(value)) return "₹0";
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
  }).format(value);
}

// ============================================================================
// Component
// ============================================================================

function InventoryManagement() {
  const [loading, setLoading] = useState(true);
  const [catalog, setCatalog] = useState<CatalogResponse | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [selectedStore, setSelectedStore] = useState<string>("1");
  const [searchQuery, setSearchQuery] = useState("");
  const [categoryFilter, setCategoryFilter] = useState<string>("All");

  // Demand Prediction States
  const [predictions, setPredictions] = useState<Record<string, ProductPredictionResult>>({});
  const [predictingProduct, setPredictingProduct] = useState<string | null>(null);
  const [predictingAll, setPredictingAll] = useState(false);
  const [activeEvidence, setActiveEvidence] = useState<ProductPredictionResult | null>(null);
  const [evidenceDialogOpen, setEvidenceDialogOpen] = useState(false);

  // Dataset Operation States
  const [syncingDataset, setSyncingDataset] = useState(false);
  const [resettingData, setResettingData] = useState(false);
  const [uploadingCsv, setUploadingCsv] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Daily Sale / Restock Entry States
  const [dailySaleOpen, setDailySaleOpen] = useState(false);
  const [stockMovementMode, setStockMovementMode] = useState<"sale" | "restock">("sale");
  const [dailySaleDate, setDailySaleDate] = useState(() => new Date().toISOString().slice(0, 10));
  const [dailySaleEntries, setDailySaleEntries] = useState<{ sku: string; qty: string }[]>([{ sku: "", qty: "" }]);
  const [submittingDailySale, setSubmittingDailySale] = useState(false);

  // Local draft purchase orders created by user
  const [draftPOs, setDraftPOs] = useState<PurchaseOrderItem[]>([]);

  // Fetch authorization headers safely
  const getAuthHeaders = useCallback(async (): Promise<Record<string, string>> => {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    try {
      const token = await auth.currentUser?.getIdToken();
      if (token) {
        headers["Authorization"] = `Bearer ${token}`;
      }
    } catch {
      // In development mode, fallback without token
    }
    return headers;
  }, []);

  // Fetch model metadata
  const fetchModelInfo = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/inventory/model-info`);
      if (res.ok) {
        const data: ModelInfo = await res.json();
        setModelInfo(data);
      }
    } catch (err) {
      console.warn("Failed to retrieve model info:", err);
    }
  }, []);

  // Fetch live inventory catalog
  const fetchCatalog = useCallback(async (storeId = selectedStore) => {
    setLoading(true);
    try {
      const headers = await getAuthHeaders();
      const queryParam = storeId === "all" ? "all" : storeId;
      const res = await fetch(`${API_BASE_URL}/inventory/catalog?store_id=${queryParam}`, {
        method: "GET",
        headers,
      });

      if (!res.ok) {
        throw new Error(`Failed to load inventory catalog (HTTP ${res.status})`);
      }

      const data: CatalogResponse = await res.json();
      setCatalog(data);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Failed to load inventory";
      toast.error(msg);
      setCatalog({
        has_data: false,
        summary: { total_skus: 0, low_stock_count: 0, total_valuation: 0, avg_cover_days: 0 },
        products: [],
        stock_movements: [],
        suppliers: [],
        purchase_orders: [],
      });
    } finally {
      setLoading(false);
    }
  }, [getAuthHeaders, selectedStore]);

  useEffect(() => {
    void fetchModelInfo();
    void fetchCatalog(selectedStore);
  }, [fetchCatalog, fetchModelInfo, selectedStore]);

  // Load Verified Store Inventory Dataset (73,000 records)
  const handleLoadVerifiedDataset = async () => {
    setSyncingDataset(true);
    try {
      const headers = await getAuthHeaders();
      const res = await fetch(`${API_BASE_URL}/inventory/load-dataset`, {
        method: "POST",
        headers,
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => null);
        throw new Error(errJson?.detail || "Failed to load verified inventory dataset");
      }

      const resData = await res.json();
      toast.success(
        `Loaded verified inventory dataset: ${resData.total_records.toLocaleString()} records across ${resData.products_count} products and ${resData.stores_count} stores.`,
      );
      await fetchCatalog(selectedStore);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Dataset loading failed");
    } finally {
      setSyncingDataset(false);
    }
  };

  // Upload Custom Inventory CSV
  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setUploadingCsv(true);
    try {
      const formData = new FormData();
      formData.append("file", file);

      const token = await auth.currentUser?.getIdToken().catch(() => null);
      const headers: Record<string, string> = {};
      if (token) headers["Authorization"] = `Bearer ${token}`;

      const res = await fetch(`${API_BASE_URL}/inventory/upload-csv`, {
        method: "POST",
        headers,
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => null);
        throw new Error(errData?.detail || `Upload failed with HTTP ${res.status}`);
      }

      const result = await res.json();
      toast.success(
        `Successfully uploaded "${file.name}": ${result.total_records.toLocaleString()} records parsed.`,
      );
      await fetchCatalog(selectedStore);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to upload inventory CSV");
    } finally {
      setUploadingCsv(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  // Reset inventory data to test and verify honest empty states
  const handleResetData = async () => {
    setResettingData(true);
    try {
      const headers = await getAuthHeaders();
      const res = await fetch(`${API_BASE_URL}/inventory/reset`, {
        method: "DELETE",
        headers,
      });

      if (res.ok) {
        setPredictions({});
        setDraftPOs([]);
        toast.info("Inventory dataset cleared. Catalog is now in empty state.");
        await fetchCatalog(selectedStore);
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to reset data");
    } finally {
      setResettingData(false);
    }
  };

  // Predict Demand for a specific product
  const handlePredictDemand = async (product: ProductItem) => {
    setPredictingProduct(product.product_id);
    try {
      const headers = await getAuthHeaders();
      const res = await fetch(`${API_BASE_URL}/inventory/predict-product`, {
        method: "POST",
        headers,
        body: JSON.stringify({
          product_id: product.product_id,
          store_id: product.store_id || selectedStore || "1",
        }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => null);
        throw new Error(
          errData?.detail || `Prediction failed for ${product.name} (HTTP ${res.status})`,
        );
      }

      const result: ProductPredictionResult = await res.json();
      setPredictions((prev) => ({
        ...prev,
        [product.product_id]: result,
      }));

      toast.success(
        `Demand predicted for ${product.name}: ${result.predicted_demand.toFixed(1)} units on ${result.target_date}.`,
      );
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to predict demand");
    } finally {
      setPredictingProduct(null);
    }
  };

  // Batch predict demand for all products
  const handlePredictAll = async () => {
    if (!catalog?.products || catalog.products.length === 0) {
      toast.warning("No products loaded to predict demand.");
      return;
    }
    setPredictingAll(true);
    try {
      const headers = await getAuthHeaders();
      const updated: Record<string, ProductPredictionResult> = {};
      const products = catalog.products;
      const chunkSize = 5;
      for (let i = 0; i < products.length; i += chunkSize) {
        const chunk = products.slice(i, i + chunkSize);
        await Promise.all(
          chunk.map(async (product) => {
            try {
              const res = await fetch(`${API_BASE_URL}/inventory/predict-product`, {
                method: "POST",
                headers,
                body: JSON.stringify({
                  product_id: product.product_id,
                  store_id: product.store_id || selectedStore || "1",
                }),
              });
              if (res.ok) {
                const data: ProductPredictionResult = await res.json();
                updated[product.product_id] = data;
              }
            } catch {
              // ignore individual failure
            }
          }),
        );
      }

      setPredictions((prev) => ({ ...prev, ...updated }));
      toast.success(
        `Generated demand forecasts for ${Object.keys(updated).length} products with replenishment advice.`,
      );
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to run batch predictions");
    } finally {
      setPredictingAll(false);
    }
  };

  // Auto-generate Purchase Orders for all products below threshold
  const handleAutoCreatePOs = () => {
    if (!catalog || !catalog.products || catalog.products.length === 0) {
      toast.warning("No products available to generate purchase orders.");
      return;
    }

    const lowStockItems = catalog.products.filter(
      (p) => p.current_stock < p.reorder_threshold,
    );

    if (lowStockItems.length === 0) {
      toast.info("All products are currently above their reorder thresholds. No POs required!");
      return;
    }

    const newPOs: PurchaseOrderItem[] = lowStockItems.map((item, idx) => {
      const pred = predictions[item.product_id];
      const recommendedQty = pred?.replenishment?.recommended_reorder_qty
        ? pred.replenishment.recommended_reorder_qty
        : Math.max(50, Math.ceil((item.reorder_threshold - item.current_stock) * 1.5));

      return {
        id: `DRAFT-PO-${Date.now().toString().slice(-4)}-${idx + 1}`,
        supplier: item.supplier,
        sku: item.sku,
        product_name: item.name,
        qty: recommendedQty,
        value: roundNum(recommendedQty * item.unit_price),
        status: "Draft",
      };
    });

    setDraftPOs((prev) => [...newPOs, ...prev]);
    toast.success(
      `Created ${newPOs.length} draft purchase orders for items below threshold. Check the Purchase Orders tab.`,
    );
  };

  // Round helper
  function roundNum(val: number): number {
    return Math.round(val * 100) / 100;
  }

  // Submit daily sales entries
  const handleSubmitDailySale = async () => {
    const valid = dailySaleEntries.filter((e) => e.sku.trim() && Number(e.qty) > 0);
    if (valid.length === 0) {
      toast.warning("Enter at least one SKU and quantity.");
      return;
    }
    setSubmittingDailySale(true);
    try {
      const headers = await getAuthHeaders();
      const res = await fetch(`${API_BASE_URL}/inventory/record-sales`, {
        method: "POST",
        headers,
        body: JSON.stringify({
          date: dailySaleDate,
          action: stockMovementMode,
          store_id: selectedStore,
          sales: valid.map((e) => ({
            sku: e.sku.trim().toUpperCase(),
            qty_sold: Number(e.qty),
            action: stockMovementMode,
          })),
        }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => null);
        throw new Error(err?.detail || `Failed (HTTP ${res.status})`);
      }

      if (stockMovementMode === "restock") {
        toast.success(`Successfully added stock for ${valid.length} product(s) on ${dailySaleDate}.`);
      } else {
        toast.success(`Recorded ${valid.length} sale(s) for ${dailySaleDate}. Catalog refreshed.`);
      }
      setDailySaleOpen(false);
      setDailySaleEntries([{ sku: "", qty: "" }]);
      await fetchCatalog(selectedStore);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to record stock update");
    } finally {
      setSubmittingDailySale(false);
    }
  };

  // Filtered Products
  const categories = useMemo(() => {
    if (!catalog?.products) return ["All"];
    const cats = Array.from(new Set(catalog.products.map((p) => p.category)));
    return ["All", ...cats];
  }, [catalog?.products]);

  const filteredProducts = useMemo(() => {
    if (!catalog?.products) return [];
    return catalog.products.filter((p) => {
      const matchesSearch =
        searchQuery === "" ||
        p.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        p.sku.toLowerCase().includes(searchQuery.toLowerCase()) ||
        p.product_id.includes(searchQuery);

      const matchesCat =
        categoryFilter === "All" || p.category === categoryFilter;

      return matchesSearch && matchesCat;
    });
  }, [catalog?.products, searchQuery, categoryFilter]);

  // Combined Purchase Orders (Backend + Drafts)
  const allPurchaseOrders = useMemo(() => {
    const existing = catalog?.purchase_orders || [];
    return [...draftPOs, ...existing];
  }, [catalog?.purchase_orders, draftPOs]);

  return (
    <>
      {/* Hidden File Input for Custom Inventory CSV */}
      <input
        type="file"
        ref={fileInputRef}
        accept=".csv"
        className="hidden"
        onChange={(e) => void handleFileUpload(e)}
      />

      <PageHeader
        eyebrow="04 · Inventory"
        title="Inventory Management"
        description="Real-time catalog stock levels, supplier lead times, purchase orders, and AI demand forecasting powered by raj_business_demand_model.joblib."
        actions={
          <div className="flex flex-wrap items-center gap-2">
            {/* Store Selector */}
            <div className="flex items-center gap-1.5 rounded-lg border border-border bg-card px-2.5 py-1.5 text-xs shadow-xs">
              <span className="font-medium text-muted-foreground">Store:</span>
              <select
                value={selectedStore}
                onChange={(e) => setSelectedStore(e.target.value)}
                className="bg-transparent font-semibold text-foreground focus:outline-hidden"
              >
                <option value="1">Store 1 (Primary)</option>
                <option value="2">Store 2</option>
                <option value="3">Store 3</option>
                <option value="4">Store 4</option>
                <option value="5">Store 5</option>
                <option value="all">All Stores</option>
              </select>
            </div>

            {/* Sync / Load Dataset Button */}
            <Button
              variant="outline"
              size="sm"
              disabled={syncingDataset || loading}
              onClick={() => void handleLoadVerifiedDataset()}
              title="Loads the verified store inventory dataset (73,000 records)"
            >
              <Database className="size-3.5" />
              {syncingDataset ? "Syncing..." : "Sync Verified Dataset"}
            </Button>

            {/* Upload Custom CSV */}
            <Button
              variant="outline"
              size="sm"
              disabled={uploadingCsv || loading}
              onClick={() => fileInputRef.current?.click()}
              title="Upload your own inventory CSV"
            >
              <UploadCloud className="size-3.5" />
              {uploadingCsv ? "Uploading..." : "Upload CSV"}
            </Button>

            {/* Add Stock (Restock In) */}
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                setStockMovementMode("restock");
                setDailySaleOpen(true);
              }}
              title="Add incoming stock from suppliers or replenishment"
            >
              <Plus className="size-3.5 text-success" />
              Add Stock
            </Button>

            {/* Record Daily Sale (Stock Out) */}
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                setStockMovementMode("sale");
                setDailySaleOpen(true);
              }}
              title="Manually record today's sales quantities per SKU"
            >
              <ClipboardList className="size-3.5" />
              Record Daily Sale
            </Button>

            {/* Auto Create POs */}
            <Button
              size="sm"
              disabled={!catalog?.has_data || catalog.summary.low_stock_count === 0}
              onClick={handleAutoCreatePOs}
            >
              <Truck className="size-3.5" />
              Auto-create POs
              {catalog && catalog.summary.low_stock_count > 0 && (
                <span className="ml-1 rounded-full bg-primary-foreground/20 px-1.5 py-0.2 text-[10px] font-bold">
                  {catalog.summary.low_stock_count}
                </span>
              )}
            </Button>

            {/* Reset / Empty State Tester */}
            {catalog?.has_data && (
              <Button
                variant="ghost"
                size="sm"
                disabled={resettingData}
                onClick={() => void handleResetData()}
                title="Clear loaded dataset to inspect honest empty state"
                className="text-muted-foreground hover:text-destructive"
              >
                <Trash2 className="size-3.5" />
              </Button>
            )}
          </div>
        }
      />

      {/* Model Contract & ML Telemetry Banner */}
      {modelInfo && (
        <div className="surface-panel relative overflow-hidden border border-primary/20 bg-gradient-to-r from-primary/5 via-primary/10 to-transparent p-4.5">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <ShieldCheck className="size-4.5 text-primary" />
                <span className="text-xs font-bold uppercase tracking-wider text-primary">
                  Trained ML Demand Engine
                </span>
                <Badge variant="secondary" className="font-mono text-[10px]">
                  {modelInfo.model_file}
                </Badge>
              </div>
              <p className="text-sm font-medium text-foreground">
                Model: <span className="font-semibold text-primary">{modelInfo.model_type}</span> ·
                Forecasting Horizon:{" "}
                <span className="font-semibold">
                  {modelInfo.forecast_horizon_days} Day (Next-Day Forecast)
                </span>{" "}
                · Trained Through:{" "}
                <span className="font-mono text-xs">{modelInfo.trained_through}</span>
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-3 text-xs">
              {modelInfo.test_metrics.testing_r2 !== undefined && (
                <div className="rounded-md border border-border/60 bg-background/80 px-2.5 py-1 backdrop-blur-xs">
                  <span className="text-muted-foreground">Test R²: </span>
                  <span className="font-bold text-success">
                    {(modelInfo.test_metrics.testing_r2 * 100).toFixed(1)}%
                  </span>
                </div>
              )}
              {modelInfo.test_metrics.test_mae !== undefined && (
                <div className="rounded-md border border-border/60 bg-background/80 px-2.5 py-1 backdrop-blur-xs">
                  <span className="text-muted-foreground">Test MAE: </span>
                  <span className="font-bold text-foreground">
                    {modelInfo.test_metrics.test_mae.toFixed(2)} units
                  </span>
                </div>
              )}
              {modelInfo.test_metrics.test_rmse !== undefined && (
                <div className="rounded-md border border-border/60 bg-background/80 px-2.5 py-1 backdrop-blur-xs">
                  <span className="text-muted-foreground">Test RMSE: </span>
                  <span className="font-bold text-foreground">
                    {modelInfo.test_metrics.test_rmse.toFixed(2)}
                  </span>
                </div>
              )}
            </div>
          </div>

          <div className="mt-3 flex flex-wrap items-center gap-1.5 border-t border-border/50 pt-2.5 text-[11px] text-muted-foreground">
            <span className="font-semibold text-foreground">Required 10 Model Features:</span>
            {modelInfo.feature_cols.map((feature) => (
              <span
                key={feature}
                className="inline-flex items-center rounded-sm bg-muted/80 px-1.5 py-0.5 font-mono text-[10px] text-foreground"
              >
                {feature}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Real Stat Cards (Derived strictly from backend data, no hardcoding) */}
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard
          label="SKUs Tracked"
          value={catalog?.has_data ? String(catalog.summary.total_skus) : "0"}
          hint={
            catalog?.has_data
              ? `${categories.length - 1} categories active`
              : "No catalog loaded"
          }
        />

        <StatCard
          label="Below Threshold"
          value={catalog?.has_data ? String(catalog.summary.low_stock_count) : "0"}
          trend={catalog && catalog.summary.low_stock_count > 0 ? "down" : "up"}
          {...(catalog && catalog.summary.low_stock_count > 0
            ? { change: `${catalog.summary.low_stock_count} require reorder` }
            : {})}
          hint={
            catalog && catalog.summary.low_stock_count === 0
              ? "All stock levels healthy"
              : "Reorder required"
          }
        />

        <StatCard
          label="Stock Valuation"
          value={
            catalog?.has_data
              ? formatCurrency(catalog.summary.total_valuation)
              : "₹0"
          }
          hint={
            catalog?.has_data
              ? `Store ${selectedStore} on-hand value`
              : "Awaiting data"
          }
        />

        <StatCard
          label="Avg Stock Cover"
          value={
            catalog?.has_data && catalog.summary.avg_cover_days > 0
              ? `${catalog.summary.avg_cover_days.toFixed(1)} days`
              : "—"
          }
          hint={
            catalog?.has_data
              ? "Weighted by actual daily velocity"
              : "Connect sales data"
          }
        />
      </div>

      {/* Product Catalog Section */}
      <Panel
        title="Product Catalog & Stock Health"
        description="Live on-hand inventory, reorder thresholds, supplier lead times, and genuine demand predictions computed from historical sales."
        actions={
          <div className="flex flex-wrap items-center gap-2">
            {/* Search Box */}
            <div className="relative w-44 sm:w-56">
              <Search className="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground" />
              <Input
                placeholder="Search SKU or name..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="h-8 pl-8 text-xs"
              />
            </div>

            {/* Category Filter */}
            <div className="flex items-center gap-1 rounded-md border border-border bg-card p-0.5 text-xs">
              <SlidersHorizontal className="ml-1.5 size-3 text-muted-foreground" />
              <select
                value={categoryFilter}
                onChange={(e) => setCategoryFilter(e.target.value)}
                className="bg-transparent px-1.5 py-1 text-xs font-medium focus:outline-hidden"
              >
                {categories.map((cat) => (
                  <option key={cat} value={cat}>
                    {cat}
                  </option>
                ))}
              </select>
            </div>

            {/* Batch Predict All Demand */}
            <Button
              variant="outline"
              size="sm"
              disabled={predictingAll || !catalog?.has_data}
              onClick={() => void handlePredictAll()}
              title="Run AI demand predictions and replenishment advice for all loaded products"
              className="h-8 text-xs gap-1.5"
            >
              <Sparkles className={`size-3.5 ${predictingAll ? "animate-spin text-primary" : "text-primary"}`} />
              {predictingAll ? "Predicting All…" : "Predict All Demand"}
            </Button>

            <Button
              variant="ghost"
              size="sm"
              disabled={loading}
              onClick={() => void fetchCatalog(selectedStore)}
              title="Refresh Catalog"
              className="h-8 px-2"
            >
              <RefreshCw className={`size-3.5 ${loading ? "animate-spin" : ""}`} />
            </Button>
          </div>
        }
        bodyClassName="p-0"
      >
        {/* Loading Spinner */}
        {loading && !catalog && (
          <div className="flex flex-col items-center justify-center p-16 text-center">
            <RefreshCw className="size-8 animate-spin text-primary" />
            <p className="mt-3 text-sm font-medium text-foreground">
              Connecting to backend inventory catalog...
            </p>
            <p className="text-xs text-muted-foreground">
              Verifying database records and model endpoints
            </p>
          </div>
        )}

        {/* Honest Empty State when no data is loaded */}
        {!loading && (!catalog?.has_data || catalog.products.length === 0) && (
          <div className="flex flex-col items-center justify-center p-12 text-center sm:p-16">
            <div className="rounded-full border border-border bg-muted/50 p-4">
              <Package className="size-10 text-muted-foreground" />
            </div>
            <h3 className="mt-4 text-base font-bold text-foreground">
              No Inventory Records Loaded
            </h3>
            <p className="mx-auto mt-1.5 max-w-md text-xs leading-relaxed text-muted-foreground">
              Your product catalog is currently empty. Connect your business dataset to see
              real-time stock levels, reorder thresholds, and generate AI demand predictions using{" "}
              <code className="rounded-sm bg-muted px-1 py-0.5 font-mono text-[11px] text-foreground">
                raj_business_demand_model.joblib
              </code>
              .
            </p>

            <div className="mt-6 flex flex-wrap items-center justify-center gap-3">
              <Button
                disabled={syncingDataset}
                onClick={() => void handleLoadVerifiedDataset()}
              >
                <Database className="size-4" />
                {syncingDataset
                  ? "Loading 73,000 Records..."
                  : "Load Verified Business Dataset"}
              </Button>

              <Button
                variant="outline"
                disabled={uploadingCsv}
                onClick={() => fileInputRef.current?.click()}
              >
                <UploadCloud className="size-4" />
                Upload Custom CSV
              </Button>
            </div>

            <div className="mt-6 rounded-lg border border-border/60 bg-muted/30 p-3 text-left text-xs text-muted-foreground">
              <p className="font-semibold text-foreground">Dataset Column Requirements:</p>
              <p className="mt-1 font-mono text-[11px]">
                Date, Store_ID, Product_ID, Units_Sold, Inventory_Level, Price, Category, Supplier_Lead_Time_Days
              </p>
            </div>
          </div>
        )}

        {/* Live Product Table */}
        {!loading && catalog?.has_data && catalog.products.length > 0 && (
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-24">SKU</TableHead>
                  <TableHead>Product / Category</TableHead>
                  <TableHead>Store & Region</TableHead>
                  <TableHead className="w-48">Stock vs Threshold</TableHead>
                  <TableHead className="text-right">Unit Price</TableHead>
                  <TableHead>Supplier & Lead Time</TableHead>
                  <TableHead className="min-w-64">AI Demand Prediction</TableHead>
                </TableRow>
              </TableHeader>

              <TableBody>
                {filteredProducts.map((product) => {
                  const isLow = product.current_stock < product.reorder_threshold;
                  const predResult = predictions[product.product_id];
                  const isPredicting = predictingProduct === product.product_id;

                  const progressVal = Math.min(
                    (product.current_stock / (product.reorder_threshold * 2)) * 100,
                    100,
                  );

                  return (
                    <TableRow key={product.product_id} className="hover:bg-muted/40">
                      {/* SKU */}
                      <TableCell className="font-mono text-xs font-semibold text-foreground">
                        {product.sku}
                      </TableCell>

                      {/* Product Name & Category */}
                      <TableCell>
                        <div className="font-medium text-foreground">{product.name}</div>
                        <div className="mt-0.5 flex items-center gap-1.5">
                          <Badge variant="outline" className="text-[10px] font-normal">
                            {product.category}
                          </Badge>
                          <span className="text-[11px] text-muted-foreground">
                            ID: {product.product_id}
                          </span>
                        </div>
                      </TableCell>

                      {/* Store & Region */}
                      <TableCell>
                        <div className="text-xs font-semibold text-foreground">
                          Store {product.store_id}
                        </div>
                        <div className="text-[11px] text-muted-foreground">
                          {product.region}
                        </div>
                      </TableCell>

                      {/* Stock vs Threshold */}
                      <TableCell>
                        <div className="space-y-1.5">
                          <div className="flex items-center justify-between text-xs">
                            <span
                              className={
                                isLow
                                  ? "flex items-center gap-1 font-bold text-destructive"
                                  : "font-semibold text-foreground"
                              }
                            >
                              {isLow && <AlertTriangle className="size-3" />}
                              {product.current_stock} units
                            </span>

                            <span className="text-[11px] text-muted-foreground">
                              threshold: {product.reorder_threshold}
                            </span>
                          </div>

                          <Progress
                            value={progressVal}
                            className={`h-1.5 ${
                              isLow
                                ? "[&>div]:bg-destructive"
                                : progressVal < 60
                                  ? "[&>div]:bg-warning"
                                  : "[&>div]:bg-success"
                            }`}
                          />

                          {product.units_ordered > 0 && (
                            <div className="flex items-center gap-1 text-[10px] font-medium text-primary">
                              <Truck className="size-3" />
                              <span>+{product.units_ordered} units on order</span>
                            </div>
                          )}
                        </div>
                      </TableCell>

                      {/* Unit Price */}
                      <TableCell className="text-right font-mono text-xs font-medium tabular-nums text-foreground">
                        {formatCurrency(product.unit_price)}
                      </TableCell>

                      {/* Supplier & Lead Time */}
                      <TableCell>
                        <div className="text-xs font-medium text-foreground">
                          {product.supplier}
                        </div>
                        <div className="mt-0.5 flex items-center gap-1 text-[11px] text-muted-foreground">
                          <span className="inline-block size-1.5 rounded-full bg-primary" />
                          <span>Lead time: {product.supplier_lead_time_days} days</span>
                        </div>
                      </TableCell>

                      {/* AI Demand Prediction & Replenishment Advice */}
                      <TableCell>
                        <div className="flex flex-col items-start gap-1.5">
                          {predResult ? (
                            <div className="w-full space-y-1 rounded-md border border-border bg-card p-2 shadow-2xs">
                              <div className="flex items-center justify-between gap-2">
                                <StatusPill tone="success">
                                  <Sparkles className="size-3" />
                                  <span className="font-bold">
                                    {predResult.predicted_demand.toFixed(1)} units
                                  </span>
                                </StatusPill>

                                <span className="text-[10px] font-medium text-muted-foreground">
                                  for {predResult.target_date}
                                </span>
                              </div>

                              {/* Replenishment Suggestion Badge */}
                              <div className="flex items-center justify-between gap-1 pt-1">
                                {predResult.replenishment.status === "sufficient" ? (
                                  <span className="inline-flex items-center gap-1 text-[11px] font-medium text-success">
                                    <CheckCircle2 className="size-3" />
                                    Healthy cover (~
                                    {predResult.replenishment.days_until_stockout.toFixed(0)}d)
                                  </span>
                                ) : (
                                  <span className="inline-flex items-center gap-1 text-[11px] font-bold text-destructive">
                                    <AlertTriangle className="size-3" />
                                    Reorder +
                                    {predResult.replenishment.recommended_reorder_qty} units
                                  </span>
                                )}

                                <button
                                  type="button"
                                  onClick={() => {
                                    setActiveEvidence(predResult);
                                    setEvidenceDialogOpen(true);
                                  }}
                                  className="inline-flex items-center text-[10px] font-semibold text-primary underline underline-offset-2 hover:opacity-80"
                                >
                                  Evidence
                                  <ChevronRight className="size-2.5" />
                                </button>
                              </div>

                              {/* Projected Daily Revenue connected to Sales module */}
                              <div className="flex items-center justify-between border-t border-border/40 pt-1 text-[11px] text-muted-foreground">
                                <span>Est. Daily Rev:</span>
                                <span className="font-semibold text-foreground">
                                  ₹{(predResult.predicted_demand * product.unit_price).toLocaleString("en-IN", { maximumFractionDigits: 0 })}
                                </span>
                              </div>
                            </div>
                          ) : (
                            <Button
                              variant="outline"
                              size="sm"
                              disabled={isPredicting}
                              onClick={() => void handlePredictDemand(product)}
                              className="h-7 text-xs"
                            >
                              <Sparkles className="size-3" />
                              {isPredicting ? "Predicting..." : "Predict Demand"}
                            </Button>
                          )}
                        </div>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </div>
        )}
      </Panel>

      {/* Tabs: Movements, Suppliers, Purchase Orders */}
      <Tabs defaultValue="movements">
        <TabsList>
          <TabsTrigger value="movements">
            <Layers className="size-3.5" />
            Stock Movements
            {catalog?.stock_movements && catalog.stock_movements.length > 0 && (
              <span className="ml-1.5 text-[10px] opacity-75">
                ({catalog.stock_movements.length})
              </span>
            )}
          </TabsTrigger>

          <TabsTrigger value="suppliers">
            <Package className="size-3.5" />
            Suppliers
            {catalog?.suppliers && catalog.suppliers.length > 0 && (
              <span className="ml-1.5 text-[10px] opacity-75">
                ({catalog.suppliers.length})
              </span>
            )}
          </TabsTrigger>

          <TabsTrigger value="pos">
            <Truck className="size-3.5" />
            Purchase Orders
            {allPurchaseOrders.length > 0 && (
              <span className="ml-1.5 text-[10px] opacity-75">
                ({allPurchaseOrders.length})
              </span>
            )}
          </TabsTrigger>
        </TabsList>

        {/* Tab 1: Stock Movements */}
        <TabsContent value="movements" className="mt-4">
          <Panel
            title="Recent Stock Movements"
            description="Verified audit log of customer sales (Out) and inbound supplier shipments (In)."
            bodyClassName="p-0"
          >
            {(!catalog?.has_data || catalog.stock_movements.length === 0) ? (
              <div className="p-8 text-center text-xs text-muted-foreground">
                No stock movements logged. Movements will be tracked dynamically as sales and PO
                deliveries occur.
              </div>
            ) : (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Movement ID</TableHead>
                    <TableHead>SKU</TableHead>
                    <TableHead>Type</TableHead>
                    <TableHead className="text-right">Quantity</TableHead>
                    <TableHead>Reference</TableHead>
                    <TableHead>Date</TableHead>
                  </TableRow>
                </TableHeader>

                <TableBody>
                  {catalog.stock_movements.map((move) => (
                    <TableRow key={move.id}>
                      <TableCell className="font-mono text-xs font-semibold text-foreground">
                        {move.id}
                      </TableCell>
                      <TableCell className="font-mono text-xs">{move.sku}</TableCell>
                      <TableCell>
                        <StatusPill tone={move.type === "In" ? "success" : "warning"}>
                          {move.type}
                        </StatusPill>
                      </TableCell>
                      <TableCell className="text-right font-mono text-xs font-bold tabular-nums">
                        {move.type === "Out" ? `-${move.qty}` : `+${move.qty}`}
                      </TableCell>
                      <TableCell className="text-xs text-muted-foreground">
                        {move.ref}
                      </TableCell>
                      <TableCell className="font-mono text-xs text-muted-foreground">
                        {move.date}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </Panel>
        </TabsContent>

        {/* Tab 2: Suppliers */}
        <TabsContent value="suppliers" className="mt-4">
          {(!catalog?.has_data || catalog.suppliers.length === 0) ? (
            <Panel bodyClassName="p-8 text-center text-xs text-muted-foreground">
              No suppliers configured. Upload an inventory CSV or sync the verified dataset to
              populate supplier relationships.
            </Panel>
          ) : (
            <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
              {catalog.suppliers.map((supplier) => (
                <div
                  key={supplier.name}
                  className="surface-panel space-y-3 p-5 transition-shadow hover:shadow-sm"
                >
                  <div className="flex items-start justify-between">
                    <div>
                      <h4 className="text-sm font-bold text-foreground">{supplier.name}</h4>
                      <p className="text-[11px] text-muted-foreground">
                        {supplier.active_skus} active SKUs supplied
                      </p>
                    </div>
                    <Badge variant="outline" className="text-[10px]">
                      {supplier.terms}
                    </Badge>
                  </div>

                  <div className="space-y-1.5 border-t border-border pt-2 text-xs">
                    <div className="flex items-center justify-between">
                      <span className="text-muted-foreground">Delivery Lead Time:</span>
                      <span className="font-semibold text-foreground">{supplier.lead}</span>
                    </div>

                    <div className="flex items-center justify-between">
                      <span className="text-muted-foreground">On-time Fulfillment:</span>
                      <span className="font-bold text-success">{supplier.reliability}</span>
                    </div>

                    <div className="flex items-center justify-between">
                      <span className="text-muted-foreground">Inventory Managed:</span>
                      <span className="font-mono font-semibold text-foreground">
                        {supplier.spend}
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </TabsContent>

        {/* Tab 3: Purchase Orders */}
        <TabsContent value="pos" className="mt-4">
          <Panel
            title="Purchase Orders"
            description="Active inbound replenishment orders and newly generated draft POs."
            bodyClassName="p-0"
          >
            {allPurchaseOrders.length === 0 ? (
              <div className="p-8 text-center text-xs text-muted-foreground">
                No active or draft purchase orders. Use the &quot;Auto-create POs&quot; button to
                automatically generate orders for items below their safety threshold.
              </div>
            ) : (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>PO Number</TableHead>
                    <TableHead>Supplier</TableHead>
                    <TableHead>SKU / Product</TableHead>
                    <TableHead className="text-right">Quantity</TableHead>
                    <TableHead className="text-right">Order Value</TableHead>
                    <TableHead>Status</TableHead>
                  </TableRow>
                </TableHeader>

                <TableBody>
                  {allPurchaseOrders.map((po) => (
                    <TableRow key={po.id}>
                      <TableCell className="font-mono text-xs font-semibold text-foreground">
                        {po.id}
                      </TableCell>
                      <TableCell className="text-xs font-medium">{po.supplier}</TableCell>
                      <TableCell>
                        <span className="font-mono text-xs font-semibold">{po.sku}</span>
                        <span className="ml-1.5 text-[11px] text-muted-foreground">
                          {po.product_name}
                        </span>
                      </TableCell>
                      <TableCell className="text-right font-mono text-xs tabular-nums">
                        {po.qty} units
                      </TableCell>
                      <TableCell className="text-right font-mono text-xs font-semibold tabular-nums text-foreground">
                        {formatCurrency(po.value)}
                      </TableCell>
                      <TableCell>
                        <StatusPill
                          tone={
                            po.status === "In Transit"
                              ? "primary"
                              : po.status === "Confirmed"
                                ? "success"
                                : "warning"
                          }
                        >
                          {po.status}
                        </StatusPill>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </Panel>
        </TabsContent>
      </Tabs>

      {/* Model Evidence & Replenishment Calculation Dialog */}
      <Dialog open={evidenceDialogOpen} onOpenChange={setEvidenceDialogOpen}>
        <DialogContent className="max-w-2xl sm:max-w-2xl">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Sparkles className="size-4 text-primary" />
              <span>Demand Prediction Evidence & Replenishment Analysis</span>
            </DialogTitle>
            <DialogDescription>
              Exact features computed from actual historical sales and passed to{" "}
              <code className="font-mono text-xs font-semibold text-foreground">
                raj_business_demand_model.joblib
              </code>
              .
            </DialogDescription>
          </DialogHeader>

          {activeEvidence && (
            <div className="space-y-4 py-2">
              {/* Product and Forecast Summary Card */}
              <div className="rounded-lg border border-border bg-muted/40 p-3.5">
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                  <div>
                    <span className="text-[11px] text-muted-foreground">Product SKU</span>
                    <p className="font-mono text-xs font-bold text-foreground">
                      {activeEvidence.sku}
                    </p>
                  </div>
                  <div>
                    <span className="text-[11px] text-muted-foreground">Store ID</span>
                    <p className="font-mono text-xs font-bold text-foreground">
                      Store {activeEvidence.store_id}
                    </p>
                  </div>
                  <div>
                    <span className="text-[11px] text-muted-foreground">Forecast Horizon</span>
                    <p className="text-xs font-bold text-foreground">Next-Day Forecast</p>
                  </div>
                  <div>
                    <span className="text-[11px] text-muted-foreground">Target Date</span>
                    <p className="font-mono text-xs font-bold text-primary">
                      {activeEvidence.target_date}
                    </p>
                  </div>
                </div>

                <div className="mt-3 flex items-center justify-between border-t border-border pt-2 text-xs">
                  <span className="text-muted-foreground">Historical records evaluated:</span>
                  <span className="font-semibold text-foreground">
                    {activeEvidence.historical_records_used} daily records (through{" "}
                    {activeEvidence.historical_end_date})
                  </span>
                </div>
              </div>

              {/* Exact Features Grid */}
              <div className="space-y-2">
                <h4 className="text-xs font-bold uppercase tracking-wider text-muted-foreground">
                  Exact Features Passed to Model
                </h4>
                <div className="grid grid-cols-2 gap-2 sm:grid-cols-5">
                  <div className="rounded-md border border-border bg-card p-2 text-center">
                    <span className="text-[10px] text-muted-foreground">lag_1 (T-1)</span>
                    <p className="mt-0.5 font-mono text-xs font-bold text-foreground">
                      {activeEvidence.features_used.lag_1} units
                    </p>
                  </div>

                  <div className="rounded-md border border-border bg-card p-2 text-center">
                    <span className="text-[10px] text-muted-foreground">lag_7 (T-7)</span>
                    <p className="mt-0.5 font-mono text-xs font-bold text-foreground">
                      {activeEvidence.features_used.lag_7} units
                    </p>
                  </div>

                  <div className="rounded-md border border-border bg-card p-2 text-center">
                    <span className="text-[10px] text-muted-foreground">lag_14 (T-14)</span>
                    <p className="mt-0.5 font-mono text-xs font-bold text-foreground">
                      {activeEvidence.features_used.lag_14} units
                    </p>
                  </div>

                  <div className="rounded-md border border-border bg-card p-2 text-center">
                    <span className="text-[10px] text-muted-foreground">lag_28 (T-28)</span>
                    <p className="mt-0.5 font-mono text-xs font-bold text-foreground">
                      {activeEvidence.features_used.lag_28} units
                    </p>
                  </div>

                  <div className="rounded-md border border-border bg-card p-2 text-center">
                    <span className="text-[10px] text-muted-foreground">rolling_mean_7</span>
                    <p className="mt-0.5 font-mono text-xs font-bold text-primary">
                      {activeEvidence.features_used.rolling_mean_7}
                    </p>
                  </div>
                </div>

                <div className="grid grid-cols-3 gap-2">
                  <div className="rounded-md border border-border bg-card p-2 text-center">
                    <span className="text-[10px] text-muted-foreground">day_of_week</span>
                    <p className="mt-0.5 font-mono text-xs font-medium text-foreground">
                      {activeEvidence.features_used.day_of_week} (Mon=0, Sun=6)
                    </p>
                  </div>

                  <div className="rounded-md border border-border bg-card p-2 text-center">
                    <span className="text-[10px] text-muted-foreground">month</span>
                    <p className="mt-0.5 font-mono text-xs font-medium text-foreground">
                      {activeEvidence.features_used.month}
                    </p>
                  </div>

                  <div className="rounded-md border border-border bg-card p-2 text-center">
                    <span className="text-[10px] text-muted-foreground">day_of_year</span>
                    <p className="mt-0.5 font-mono text-xs font-medium text-foreground">
                      Day {activeEvidence.features_used.day_of_year}
                    </p>
                  </div>
                </div>
              </div>

              {/* Replenishment Recommendation Reasoning */}
              <div
                className={`rounded-lg border p-3.5 text-xs ${
                  activeEvidence.replenishment.status === "sufficient"
                    ? "border-success/30 bg-success/5 text-foreground"
                    : "border-destructive/30 bg-destructive/5 text-foreground"
                }`}
              >
                <div className="flex items-center gap-1.5 font-bold">
                  {activeEvidence.replenishment.status === "sufficient" ? (
                    <>
                      <CheckCircle2 className="size-4 text-success" />
                      <span className="text-success">Stock Level Adequate</span>
                    </>
                  ) : (
                    <>
                      <AlertTriangle className="size-4 text-destructive" />
                      <span className="text-destructive">
                        Replenishment Recommended: Reorder{" "}
                        {activeEvidence.replenishment.recommended_reorder_qty} units
                      </span>
                    </>
                  )}
                </div>

                <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
                  {activeEvidence.replenishment.reason}
                </p>

                <div className="mt-3 grid grid-cols-3 gap-2 border-t border-border/40 pt-2 text-[11px]">
                  <div>
                    <span className="text-muted-foreground">Lead Time Demand:</span>
                    <p className="font-semibold">
                      {activeEvidence.replenishment.lead_time_demand} units
                    </p>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Target Safety Buffer:</span>
                    <p className="font-semibold">
                      {activeEvidence.replenishment.safety_target} units
                    </p>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Estimated Stockout:</span>
                    <p className="font-semibold">
                      ~{activeEvidence.replenishment.days_until_stockout.toFixed(1)} days
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}

          <DialogFooter>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setEvidenceDialogOpen(false)}
            >
              Close
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* ── Record Daily Sale Dialog ── */}
      <Dialog open={dailySaleOpen} onOpenChange={setDailySaleOpen}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              {stockMovementMode === "restock" ? (
                <>
                  <Plus className="size-4 text-success" />
                  Add Stock / Receive Inbound
                </>
              ) : (
                <>
                  <ClipboardList className="size-4 text-primary" />
                  Record Daily Sales
                </>
              )}
            </DialogTitle>
            <DialogDescription>
              {stockMovementMode === "restock"
                ? "Enter the SKU and units received from your supplier. This increments on-hand stock and logs an inbound shipment."
                : "Enter the SKU and units sold for each product today. This updates stock levels and updates demand lag features."}
            </DialogDescription>
          </DialogHeader>

          {/* Mode Switcher Toggle */}
          <div className="grid grid-cols-2 gap-1 rounded-lg border border-border bg-muted/40 p-1 text-xs">
            <button
              type="button"
              onClick={() => setStockMovementMode("sale")}
              className={`flex items-center justify-center gap-1.5 rounded-md py-1.5 font-medium transition-all ${
                stockMovementMode === "sale"
                  ? "bg-card font-bold text-foreground shadow-xs"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              <Minus className="size-3 text-warning" />
              Customer Sale (Stock Out)
            </button>
            <button
              type="button"
              onClick={() => setStockMovementMode("restock")}
              className={`flex items-center justify-center gap-1.5 rounded-md py-1.5 font-medium transition-all ${
                stockMovementMode === "restock"
                  ? "bg-card font-bold text-foreground shadow-xs"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              <Plus className="size-3 text-success" />
              Receive Inbound (Add Stock)
            </button>
          </div>

          {/* Date */}
          <div className="flex items-center gap-3 rounded-lg border border-border bg-muted/30 px-3 py-2">
            <Calendar className="size-4 text-muted-foreground" />
            <label className="text-xs font-medium text-muted-foreground">
              {stockMovementMode === "restock" ? "Restock Date" : "Sale Date"}
            </label>
            <input
              type="date"
              value={dailySaleDate}
              onChange={(e) => setDailySaleDate(e.target.value)}
              className="ml-auto rounded border border-border bg-background px-2 py-1 text-xs focus:outline-hidden focus:ring-1 focus:ring-primary"
            />
          </div>

          {/* SKU rows */}
          <div className="flex flex-col gap-2 max-h-64 overflow-y-auto pr-1">
            {dailySaleEntries.map((entry, idx) => (
              <div key={idx} className="flex items-center gap-2">
                {/* SKU dropdown from loaded products or free text */}
                {catalog?.products && catalog.products.length > 0 ? (
                  <select
                    value={entry.sku}
                    onChange={(e) => {
                      const val = e.target.value;
                      setDailySaleEntries((prev) =>
                        prev.map((item, i) => (i === idx ? { ...item, sku: val } : item))
                      );
                    }}
                    className="flex-1 rounded-md border border-border bg-background px-2 py-1.5 text-xs focus:outline-hidden focus:ring-1 focus:ring-primary"
                  >
                    <option value="">Select SKU…</option>
                    {catalog.products.map((p) => (
                      <option key={p.sku} value={p.sku}>
                        {p.sku} — {p.name}
                      </option>
                    ))}
                  </select>
                ) : (
                  <Input
                    placeholder="SKU (e.g. SKU-1001)"
                    value={entry.sku}
                    onChange={(e) => {
                      const val = e.target.value;
                      setDailySaleEntries((prev) =>
                        prev.map((item, i) => (i === idx ? { ...item, sku: val } : item))
                      );
                    }}
                    className="flex-1 h-8 text-xs"
                  />
                )}
                <Input
                  type="number"
                  min={1}
                  placeholder={stockMovementMode === "restock" ? "Qty to add" : "Qty sold"}
                  value={entry.qty}
                  onChange={(e) => {
                    const val = e.target.value;
                    setDailySaleEntries((prev) =>
                      prev.map((item, i) => (i === idx ? { ...item, qty: val } : item))
                    );
                  }}
                  className="w-24 h-8 text-xs"
                />
                {/* Remove row */}
                {dailySaleEntries.length > 1 && (
                  <button
                    onClick={() =>
                      setDailySaleEntries((prev) => prev.filter((_, i) => i !== idx))
                    }
                    className="rounded p-1 text-muted-foreground hover:text-destructive"
                  >
                    <Minus className="size-3.5" />
                  </button>
                )}
              </div>
            ))}
          </div>

          {/* Add row */}
          <button
            onClick={() =>
              setDailySaleEntries((prev) => [...prev, { sku: "", qty: "" }])
            }
            className="flex items-center gap-1.5 text-xs text-primary hover:underline"
          >
            <Plus className="size-3.5" />
            Add another SKU
          </button>

          <DialogFooter>
            <Button variant="outline" size="sm" onClick={() => setDailySaleOpen(false)}>
              Cancel
            </Button>
            <Button
              size="sm"
              disabled={submittingDailySale}
              onClick={() => void handleSubmitDailySale()}
            >
              {submittingDailySale
                ? "Saving…"
                : stockMovementMode === "restock"
                  ? "Add Stock"
                  : "Save Sales"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
export default InventoryManagement;