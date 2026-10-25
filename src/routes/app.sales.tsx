import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useState, useCallback } from "react";
import { toast } from "sonner";
import { 
  Loader2, 
  Database, 
  AlertCircle, 
  LineChart as LineChartIcon,
  RefreshCw,
  Eye,
  Trash2,
  History,
  TrendingUp,
  Calendar,
  Plus,
  Pencil
} from "lucide-react";
import { Line, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis, Legend, Area, ComposedChart } from "recharts";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { PageHeader, Panel } from "@/components/app-ui";
import { cn } from "@/lib/utils";
import { auth, getOnboardingProfileAsync, type OnboardingProfile } from "@/lib/auth";

export const Route = createFileRoute("/app/sales")({
  head: () => ({
    meta: [
      { title: "Sales Forecasting — SmartBPI" },
    ],
  }),
  component: SalesForecasting,
});

type ForecastPeriod = "30d" | "6m" | "1y";

interface ForecastRecord {
  date: string;
  predicted_sales: number;
  lower_bound: number;
  upper_bound: number;
}

interface ForecastSummary {
  historical_total_sales: number;
  historical_average_daily_sales: number;
  forecast_30_days_total: number;
  forecast_30_days_average: number;
  forecast_6_months_total: number;
  forecast_6_months_average: number;
  forecast_1_year_total: number;
  forecast_1_year_average: number;
  expected_growth_percentage: number;
  forecast_start_date: string;
  forecast_end_date: string;
}

interface ForecastResponse {
  summary: ForecastSummary;
  forecast: ForecastRecord[];
  id?: string;
  created_at?: string;
}

interface ForecastHistoryItem {
  id: string;
  user_id: string;
  forecast_days: number;
  historical_start_date: string | null;
  historical_end_date: string | null;
  historical_total_sales: number | null;
  historical_average_daily_sales: number | null;
  forecast_start_date: string;
  forecast_end_date: string;
  forecast_total: number;
  forecast_average: number;
  expected_growth_percentage: number;
  csv_source_url: string;
  forecast_records?: ForecastRecord[] | null;
  created_at: string;
}

function getForecastDateWindow(item: {
  forecast_start_date: string;
  forecast_end_date: string;
  forecast_records?: ForecastRecord[] | null;
}): { start: string; end: string } {
  if (Array.isArray(item.forecast_records) && item.forecast_records.length > 0) {
    const first = item.forecast_records[0];
    const last = item.forecast_records[item.forecast_records.length - 1];
    if (first?.date && last?.date) {
      return { start: first.date, end: last.date };
    }
  }
  return { start: item.forecast_start_date, end: item.forecast_end_date };
}

interface HistoryApiResponse {
  history: ForecastHistoryItem[];
  total: number;
  limit: number;
  offset: number;
}

interface ForecastDetailRecord {
  id: string;
  user_id: string;
  forecast_days: number;
  historical_start_date: string | null;
  historical_end_date: string | null;
  historical_total_sales: number | null;
  historical_average_daily_sales: number | null;
  forecast_start_date: string;
  forecast_end_date: string;
  forecast_total: number;
  forecast_average: number;
  expected_growth_percentage: number;
  csv_source_url: string;
  idempotency_key?: string;
  summary_metrics?: ForecastSummary;
  forecast_records: ForecastRecord[];
  created_at: string;
}

interface DailySalesItem {
  id: string;
  user_id: string;
  sale_date: string;
  daily_revenue: number;
  created_at: string;
  updated_at?: string;
}

interface DailySalesApiResponse {
  entries: DailySalesItem[];
  total: number;
  limit: number;
  offset: number;
}

function formatCurrency(value: number | null | undefined) {
  if (value === null || value === undefined || isNaN(value)) return "₹0";
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
  }).format(value);
}

function formatDate(dateStr: string | null | undefined) {
  if (!dateStr) return "—";
  try {
    const d = new Date(dateStr);
    if (isNaN(d.getTime())) return dateStr;
    return d.toLocaleDateString("en-IN", {
      year: "numeric",
      month: "short",
      day: "numeric",
    });
  } catch {
    return dateStr;
  }
}

function formatDateTime(dateStr: string | null | undefined) {
  if (!dateStr) return "—";
  try {
    const d = new Date(dateStr);
    if (isNaN(d.getTime())) return dateStr;
    return d.toLocaleDateString("en-IN", {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return dateStr;
  }
}

function formatHorizon(days: number) {
  if (days === 30) return "30 Days";
  if (days === 180) return "6 Months";
  if (days === 365) return "1 Year";
  return `${days} Days`;
}

function SalesForecasting() {
  const [profile, setProfile] = useState<OnboardingProfile | null>(null);
  const [loading, setLoading] = useState(true);
  
  const [period, setPeriod] = useState<ForecastPeriod>("30d");
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [forecast, setForecast] = useState<ForecastResponse | null>(null);

  // History state
  const [history, setHistory] = useState<ForecastHistoryItem[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [historyTotal, setHistoryTotal] = useState(0);

  // Daily Sales state
  const [dailySales, setDailySales] = useState<DailySalesItem[]>([]);
  const [dailySalesLoading, setDailySalesLoading] = useState(false);
  const [dailySalesError, setDailySalesError] = useState<string | null>(null);
  const [dailySalesTotal, setDailySalesTotal] = useState(0);

  // Add / Edit Daily Sales Dialog state
  const [dailySalesDialogOpen, setDailySalesDialogOpen] = useState(false);
  const [editingEntry, setEditingEntry] = useState<DailySalesItem | null>(null);
  const [inputDate, setInputDate] = useState("");
  const [inputRevenue, setInputRevenue] = useState("");
  const [isSavingDailySales, setIsSavingDailySales] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);

  // Detail Modal state
  const [detailModalOpen, setDetailModalOpen] = useState(false);
  const [detailRecord, setDetailRecord] = useState<ForecastDetailRecord | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);

  // Delete Confirmation state
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [recordToDelete, setRecordToDelete] = useState<ForecastHistoryItem | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);

  const loadHistory = useCallback(async () => {
    setHistoryLoading(true);
    setHistoryError(null);
    try {
      const apiUrl = (import.meta.env["VITE_FORECAST_API_URL"] as string | undefined) || "http://localhost:8000";
      const token = await auth.currentUser?.getIdToken();
      if (!token) {
        setHistoryError("Authentication required. Please sign in to view your forecast history.");
        setHistoryLoading(false);
        return;
      }
      
      const res = await fetch(`${apiUrl}/forecast/history?limit=20&offset=0`, {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${token}`
        }
      });

      if (res.status === 401) {
        setHistoryError("Session expired or unauthorized. Please re-authenticate.");
        return;
      }

      if (!res.ok) {
        const errData = await res.json().catch(() => null);
        throw new Error(errData?.detail || `Server error (${res.status}) while retrieving history.`);
      }

      const data: HistoryApiResponse = await res.json();
      setHistory(data.history || []);
      setHistoryTotal(data.total !== undefined ? data.total : (data.history ? data.history.length : 0));
    } catch (err: any) {
      console.error("Failed to load forecast history:", err);
      setHistoryError(err.message || "Failed to load forecast history.");
    } finally {
      setHistoryLoading(false);
    }
  }, []);

  const loadDailySales = useCallback(async () => {
    setDailySalesLoading(true);
    setDailySalesError(null);
    try {
      const apiUrl = (import.meta.env["VITE_FORECAST_API_URL"] as string | undefined) || "http://localhost:8000";
      const token = await auth.currentUser?.getIdToken();
      if (!token) {
        setDailySalesLoading(false);
        return;
      }

      const res = await fetch(`${apiUrl}/sales/daily?limit=50&offset=0`, {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${token}`
        }
      });

      if (res.status === 401) {
        setDailySalesError("Session expired or unauthorized. Please re-authenticate.");
        return;
      }

      if (!res.ok) {
        const errData = await res.json().catch(() => null);
        throw new Error(errData?.detail || `Server error (${res.status}) while retrieving daily sales.`);
      }

      const data: DailySalesApiResponse = await res.json();
      setDailySales(data.entries || []);
      setDailySalesTotal(data.total !== undefined ? data.total : (data.entries ? data.entries.length : 0));
    } catch (err: any) {
      console.error("Failed to load daily sales:", err);
      setDailySalesError(err.message || "Failed to load daily sales.");
    } finally {
      setDailySalesLoading(false);
    }
  }, []);

  useEffect(() => {
    const unsubscribe = auth.onAuthStateChanged(async (user: any) => {
      if (user) {
        const p = await getOnboardingProfileAsync(user.uid);
        setProfile(p);
        await Promise.all([loadHistory(), loadDailySales()]);
      } else {
        setProfile(null);
        setHistory([]);
        setDailySales([]);
      }
      setLoading(false);
    });
    return () => unsubscribe();
  }, [loadHistory, loadDailySales]);

  const handleOpenAddDailySales = () => {
    setEditingEntry(null);
    const todayStr = new Date().toISOString().split("T")[0] ?? "";
    setInputDate(todayStr);
    setInputRevenue("");
    setFormError(null);
    setDailySalesDialogOpen(true);
  };

  const handleOpenEditDailySales = (item: DailySalesItem) => {
    setEditingEntry(item);
    setInputDate(item.sale_date);
    setInputRevenue(item.daily_revenue.toString());
    setFormError(null);
    setDailySalesDialogOpen(true);
  };

  const handleSaveDailySales = async () => {
    setFormError(null);

    if (!inputDate || !inputDate.trim()) {
      setFormError("Please select a valid sales date.");
      return;
    }

    if (!inputRevenue || !inputRevenue.trim()) {
      setFormError("Please enter daily revenue.");
      return;
    }

    const parsedRev = parseFloat(inputRevenue);
    if (isNaN(parsedRev) || !isFinite(parsedRev)) {
      setFormError("Daily revenue must be a valid finite number.");
      return;
    }

    if (parsedRev < 0) {
      setFormError("Daily revenue cannot be negative.");
      return;
    }

    setIsSavingDailySales(true);
    try {
      const apiUrl = (import.meta.env["VITE_FORECAST_API_URL"] as string | undefined) || "http://localhost:8000";
      const token = await auth.currentUser?.getIdToken();
      if (!token) {
        throw new Error("Authentication required. Please sign in to save daily sales.");
      }

      let res: Response;
      if (editingEntry) {
        res = await fetch(`${apiUrl}/sales/daily/${editingEntry.id}`, {
          method: "PUT",
          headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${token}`
          },
          body: JSON.stringify({
            daily_revenue: parsedRev,
            sale_date: inputDate
          })
        });
      } else {
        res = await fetch(`${apiUrl}/sales/daily`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${token}`
          },
          body: JSON.stringify({
            sale_date: inputDate,
            daily_revenue: parsedRev
          })
        });
      }

      if (res.status === 409) {
        const errData = await res.json().catch(() => null);
        throw new Error(errData?.detail || `A sales entry for date '${inputDate}' already exists.`);
      }

      if (!res.ok) {
        const errData = await res.json().catch(() => null);
        throw new Error(errData?.detail || `API Error: ${res.status}`);
      }

      toast.success(editingEntry ? "Daily sales entry updated successfully." : "Daily sales entry saved successfully.");
      setDailySalesDialogOpen(false);
      setEditingEntry(null);
      setInputDate("");
      setInputRevenue("");
      loadDailySales();
    } catch (err: any) {
      console.error("Save daily sales error:", err);
      setFormError(err.message || "Failed to save daily sales entry.");
    } finally {
      setIsSavingDailySales(false);
    }
  };

  const handleGenerate = async () => {
    setError(null);
    setIsGenerating(true);
    setForecast(null);
    
    try {
      const apiUrl = (import.meta.env["VITE_FORECAST_API_URL"] as string | undefined) || "http://localhost:8000";
      const token = await auth.currentUser?.getIdToken();
      if (!token) {
        throw new Error("Authentication required. Please sign in to generate a forecast.");
      }
      
      const payload: Record<string, any> = {
        forecast_days: period === "30d" ? 30 : period === "6m" ? 180 : 365,
        use_inventory_data: true,
      };
      if (profile?.servicesData?.salesCsvUrl) {
        payload["sales_csv_url"] = profile.servicesData.salesCsvUrl;
      }

      const res = await fetch(`${apiUrl}/forecast`, {
        method: "POST",
        headers: { 
          "Content-Type": "application/json",
          "Authorization": `Bearer ${token}`
        },
        body: JSON.stringify(payload)
      });
      
      if (!res.ok) {
        const errData = await res.json().catch(() => null);
        throw new Error(errData?.detail || `API Error: ${res.status}`);
      }
      
      const data = await res.json();
      const records = data.forecast;
      setForecast({ 
        summary: data.summary, 
        forecast: records,
        id: data.id,
        created_at: data.created_at
      });
      toast.success("Forecast generated and saved successfully!");
      await loadHistory();
    } catch (err: any) {
      console.error(err);
      const msg = err.message || "Failed to generate forecast.";
      setError(msg);
      toast.error(msg);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleViewDetail = async (id: string) => {
    setDetailModalOpen(true);
    setDetailLoading(true);
    setDetailError(null);
    setDetailRecord(null);

    try {
      const apiUrl = (import.meta.env["VITE_FORECAST_API_URL"] as string | undefined) || "http://localhost:8000";
      const token = await auth.currentUser?.getIdToken();
      if (!token) {
        throw new Error("Authentication required to view forecast details.");
      }

      const res = await fetch(`${apiUrl}/forecast/${id}`, {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${token}`
        }
      });

      if (res.status === 404) {
        throw new Error("Forecast record not found or already deleted.");
      }

      if (res.status === 401) {
        throw new Error("Unauthorized to access this forecast record.");
      }

      if (!res.ok) {
        const errData = await res.json().catch(() => null);
        throw new Error(errData?.detail || `Server error (${res.status}) while retrieving forecast.`);
      }

      const data: ForecastDetailRecord = await res.json();
      setDetailRecord(data);
    } catch (err: any) {
      console.error("Failed to load forecast detail:", err);
      setDetailError(err.message || "Failed to load forecast details.");
    } finally {
      setDetailLoading(false);
    }
  };

  const handleOpenDelete = (item: ForecastHistoryItem) => {
    setRecordToDelete(item);
    setDeleteDialogOpen(true);
  };

  const handleConfirmDelete = async () => {
    if (!recordToDelete) return;
    setIsDeleting(true);
    try {
      const apiUrl = (import.meta.env["VITE_FORECAST_API_URL"] as string | undefined) || "http://localhost:8000";
      const token = await auth.currentUser?.getIdToken();
      if (!token) {
        throw new Error("Authentication required.");
      }

      const res = await fetch(`${apiUrl}/forecast/${recordToDelete.id}`, {
        method: "DELETE",
        headers: {
          "Authorization": `Bearer ${token}`
        }
      });

      if (res.status === 404) {
        throw new Error("Forecast record not found or already deleted.");
      }

      if (res.status === 401) {
        throw new Error("Unauthorized to delete this forecast record.");
      }

      if (!res.ok) {
        const errData = await res.json().catch(() => null);
        throw new Error(errData?.detail || `Failed to delete record (${res.status}).`);
      }

      toast.success("Forecast record deleted successfully.");
      setDeleteDialogOpen(false);
      setRecordToDelete(null);
      loadHistory();
    } catch (err: any) {
      console.error("Delete failed:", err);
      toast.error(err.message || "Failed to delete forecast record.");
    } finally {
      setIsDeleting(false);
    }
  };

  const csvUrl = profile?.servicesData?.salesCsvUrl;
  const fileName = csvUrl ? csvUrl.split('/').pop()?.split('?')[0] : null;
  const canForecast = true; // Always connected via Store Inventory Dataset or custom CSV/manual sales

  if (loading) {
    return <div className="p-8 flex justify-center"><Loader2 className="animate-spin text-primary size-8" /></div>;
  }

  return (
    <>
      <PageHeader
        eyebrow="03 · Sales & Revenue"
        title="Sales Forecasting"
        description="AI-powered sales and revenue predictions seamlessly connected with your store inventory."
      />

      <div className="grid gap-6 xl:grid-cols-3">
        {/* Sales Data Status */}
        <Panel title="Sales Data Status" className="xl:col-span-1">
          <div className="flex flex-col gap-4">
            {csvUrl ? (
              <div className="bg-primary/10 border border-primary/20 p-4 rounded-lg flex items-start gap-3">
                <Database className="text-primary size-5 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-primary">Sales CSV ready</h4>
                  <p className="text-sm text-muted-foreground break-all">{fileName}</p>
                </div>
              </div>
            ) : (
              <div className="bg-primary/5 border border-primary/20 p-4 rounded-lg flex items-start gap-3">
                <Database className="text-primary size-5 mt-0.5" />
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <h4 className="font-semibold text-primary">Connected to Store Inventory</h4>
                    <Badge variant="secondary" className="text-[10px]">730 Days Sales</Badge>
                  </div>
                  <p className="text-xs text-muted-foreground leading-relaxed">
                    Aggregated store revenue across 20 SKUs is automatically synced from Inventory Management.
                  </p>
                  <div className="pt-1">
                    <Button variant="outline" size="sm" asChild className="h-7 text-xs">
                      <a href="/app/inventory">Manage SKU Catalog →</a>
                    </Button>
                  </div>
                </div>
              </div>
            )}

            {/* Quick Action: Add Daily Sales */}
            <div className="flex gap-2">
              <Button 
                variant="outline" 
                size="sm" 
                onClick={handleOpenAddDailySales}
                className="w-full gap-1.5"
              >
                <Plus className="size-4 text-primary" />
                <span>Add Daily Sales</span>
              </Button>
            </div>

            {/* Forecast Horizon */}
            <div className="mt-2">
              <label className="text-sm font-medium mb-2 block">Forecast Horizon</label>
              <div className="flex gap-2">
                <Button 
                  variant={period === "30d" ? "default" : "outline"} 
                  onClick={() => setPeriod("30d")}
                  className="flex-1"
                >
                  30 Days
                </Button>
                <Button 
                  variant={period === "6m" ? "default" : "outline"} 
                  onClick={() => setPeriod("6m")}
                  className="flex-1"
                >
                  6 Months
                </Button>
                <Button 
                  variant={period === "1y" ? "default" : "outline"} 
                  onClick={() => setPeriod("1y")}
                  className="flex-1"
                >
                  1 Year
                </Button>
              </div>
            </div>

            <Button 
              className="w-full mt-4" 
              size="lg" 
              onClick={handleGenerate}
              disabled={isGenerating || !canForecast}
            >
              {isGenerating ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Generating predictions...
                </>
              ) : (
                <>
                  <LineChartIcon className="mr-2 h-4 w-4" />
                  Generate Forecast
                </>
              )}
            </Button>

            {error && (
              <div className="text-sm text-destructive mt-2 p-2 bg-destructive/10 rounded">
                {error}
              </div>
            )}
          </div>
        </Panel>

        {/* Current Active Forecast Summary */}
        <Panel title="AI Forecast Summary" className="xl:col-span-2">
          {!forecast && !isGenerating ? (
            <div className="flex flex-col items-center justify-center h-full min-h-[250px] text-center text-muted-foreground">
              <LineChartIcon className="size-12 mb-4 opacity-20" />
              <h3 className="text-lg font-medium text-foreground">Ready to forecast your sales</h3>
              <p className="max-w-md mt-2">
                Historical sales data from your CSV and manual daily entries will be combined using AI sales forecasting.
              </p>
            </div>
          ) : isGenerating ? (
            <div className="flex flex-col items-center justify-center h-full min-h-[250px] text-center text-muted-foreground gap-4">
              <Loader2 className="size-8 animate-spin text-primary" />
              <div className="space-y-1">
                <p className="font-medium text-foreground animate-pulse">Analyzing your sales data...</p>
                <p className="text-sm">Training seasonal sales forecasting model...</p>
              </div>
            </div>
          ) : forecast ? (
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-1 bg-muted/30 p-4 rounded-lg">
                <p className="text-sm text-muted-foreground">Historical Average</p>
                <p className="text-2xl font-bold">{formatCurrency(forecast.summary.historical_average_daily_sales)}</p>
              </div>
              <div className="space-y-1 bg-muted/30 p-4 rounded-lg">
                <p className="text-sm text-muted-foreground">Expected Growth</p>
                <p className={cn(
                  "text-2xl font-bold",
                  (forecast.summary.expected_growth_percentage ?? 0) >= 0 ? "text-green-500" : "text-destructive"
                )}>
                  {(forecast.summary.expected_growth_percentage ?? 0) >= 0 ? "+" : ""}
                  {forecast.summary.expected_growth_percentage}%
                </p>
              </div>
              <div className="space-y-1 bg-muted/30 p-4 rounded-lg">
                <p className="text-sm text-muted-foreground">Next 30 Days</p>
                <p className="text-2xl font-bold">{formatCurrency(forecast.summary.forecast_30_days_total)}</p>
                <p className="text-xs text-muted-foreground">{formatCurrency(forecast.summary.forecast_30_days_average)} / day</p>
              </div>
              <div className="space-y-1 bg-muted/30 p-4 rounded-lg">
                <p className="text-sm text-muted-foreground">
                  {period === "1y" ? "Next 1 Year" : "Next 6 Months"}
                </p>
                <p className="text-2xl font-bold">
                  {formatCurrency(period === "1y" ? forecast.summary.forecast_1_year_total : forecast.summary.forecast_6_months_total)}
                </p>
                <p className="text-xs text-muted-foreground">
                  {formatCurrency(period === "1y" ? forecast.summary.forecast_1_year_average : forecast.summary.forecast_6_months_average)} / day
                </p>
              </div>
            </div>
          ) : null}
        </Panel>
      </div>

      {/* Generated Forecast Details & Chart */}
      {forecast && (
        <div className="space-y-6">
          <Panel title="Forecast Chart" description="Sales forecast with 95% confidence interval">
            <div className="h-[400px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={forecast.forecast} margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--color-border)" />
                  <XAxis 
                    dataKey="date" 
                    stroke="var(--color-muted-foreground)"
                    fontSize={12}
                    tickMargin={10}
                  />
                  <YAxis 
                    stroke="var(--color-muted-foreground)"
                    fontSize={12}
                    tickFormatter={(value) => `₹${value / 1000}k`}
                  />
                  <Tooltip 
                    formatter={(value: number) => formatCurrency(value)}
                    contentStyle={{ borderRadius: 8, border: "1px solid var(--color-border)" }}
                  />
                  <Legend />
                  <Area 
                    type="monotone" 
                    dataKey={["lower_bound", "upper_bound"] as any}
                    stroke="none" 
                    fill="var(--color-primary)" 
                    fillOpacity={0.15} 
                    name="95% Confidence Interval"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="predicted_sales" 
                    stroke="var(--color-primary)" 
                    strokeWidth={2}
                    dot={false}
                    name="Forecast Sales"
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </Panel>

          <Panel title="Forecast Records" description="Detailed daily predictions table" bodyClassName="p-0">
            <div className="max-h-[400px] overflow-auto">
              <Table>
                <TableHeader className="sticky top-0 bg-background z-10">
                  <TableRow>
                    <TableHead>Date</TableHead>
                    <TableHead className="text-right">Forecast Sales</TableHead>
                    <TableHead className="text-right">Lower Bound</TableHead>
                    <TableHead className="text-right">Upper Bound</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {forecast.forecast.map((row, i) => (
                    <TableRow key={i}>
                      <TableCell className="font-medium">{row.date}</TableCell>
                      <TableCell className="text-right font-bold text-primary tabular-nums">
                        {formatCurrency(row.predicted_sales)}
                      </TableCell>
                      <TableCell className="text-right text-muted-foreground tabular-nums">
                        {formatCurrency(row.lower_bound)}
                      </TableCell>
                      <TableCell className="text-right text-muted-foreground tabular-nums">
                        {formatCurrency(row.upper_bound)}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </Panel>
        </div>
      )}

      {/* Daily Sales Records Section */}
      <Panel
        title="Recorded Daily Sales"
        description="View and update daily business sales. Entries are combined with your historical sales for forecasting."
        actions={
          <div className="flex items-center gap-2">
            <Button 
              variant="outline" 
              size="sm" 
              onClick={handleOpenAddDailySales}
              className="h-8 gap-1.5"
            >
              <Plus className="size-3.5 text-primary" />
              <span>Add Entry</span>
            </Button>
            <Button 
              variant="outline" 
              size="sm" 
              onClick={() => loadDailySales()} 
              disabled={dailySalesLoading}
              className="h-8 gap-1.5"
            >
              <RefreshCw className={cn("size-3.5", dailySalesLoading && "animate-spin")} />
              <span>Refresh</span>
            </Button>
          </div>
        }
      >
        {dailySalesLoading && dailySales.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-8 text-muted-foreground gap-2">
            <Loader2 className="size-5 animate-spin text-primary" />
            <p className="text-sm">Loading daily sales...</p>
          </div>
        ) : dailySalesError ? (
          <div className="p-4 rounded-lg bg-destructive/10 border border-destructive/20 text-destructive flex items-center justify-between">
            <div className="flex items-center gap-3">
              <AlertCircle className="size-5 shrink-0" />
              <p className="text-sm">{dailySalesError}</p>
            </div>
            <Button variant="outline" size="sm" onClick={() => loadDailySales()} className="shrink-0 ml-4">
              Retry
            </Button>
          </div>
        ) : dailySales.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-8 text-center text-muted-foreground">
            <Calendar className="size-8 mb-2 opacity-30" />
            <h4 className="text-sm font-semibold text-foreground">No daily sales entries recorded yet</h4>
            <p className="text-xs max-w-sm mt-1">
              Click "Add Daily Sales" above to record daily revenue and keep your forecasting model updated.
            </p>
          </div>
        ) : (
          <div className="border rounded-lg overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Sales Date</TableHead>
                  <TableHead className="text-right">Daily Revenue</TableHead>
                  <TableHead className="text-right">Last Recorded</TableHead>
                  <TableHead className="text-center w-[90px]">Action</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {dailySales.map((entry) => (
                  <TableRow key={entry.id} className="hover:bg-muted/40 transition-colors">
                    <TableCell className="font-medium text-xs">
                      {formatDate(entry.sale_date)}
                    </TableCell>
                    <TableCell className="text-right font-bold text-primary tabular-nums text-xs">
                      {formatCurrency(entry.daily_revenue)}
                    </TableCell>
                    <TableCell className="text-right text-xs text-muted-foreground tabular-nums">
                      {formatDateTime(entry.updated_at || entry.created_at)}
                    </TableCell>
                    <TableCell className="text-center">
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7 hover:text-primary"
                        title="Edit Daily Sales"
                        onClick={() => handleOpenEditDailySales(entry)}
                      >
                        <Pencil className="size-3.5" />
                        <span className="sr-only">Edit</span>
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        )}
      </Panel>

      {/* Forecast History Section */}
      <Panel
        title="Saved Forecast History"
        description="Review previously persisted sales forecasts and projections."
        actions={
          <Button 
            variant="outline" 
            size="sm" 
            onClick={() => loadHistory()} 
            disabled={historyLoading}
            className="h-8 gap-1.5"
          >
            <RefreshCw className={cn("size-3.5", historyLoading && "animate-spin")} />
            <span>Refresh</span>
          </Button>
        }
      >
        {historyLoading && history.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-12 text-muted-foreground gap-3">
            <Loader2 className="size-6 animate-spin text-primary" />
            <p className="text-sm">Loading forecast history...</p>
          </div>
        ) : historyError ? (
          <div className="p-4 rounded-lg bg-destructive/10 border border-destructive/20 text-destructive flex items-center justify-between">
            <div className="flex items-center gap-3">
              <AlertCircle className="size-5 shrink-0" />
              <p className="text-sm">{historyError}</p>
            </div>
            <Button variant="outline" size="sm" onClick={() => loadHistory()} className="shrink-0 ml-4">
              Retry
            </Button>
          </div>
        ) : history.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-12 text-center text-muted-foreground">
            <History className="size-10 mb-3 opacity-30" />
            <h4 className="text-base font-semibold text-foreground">No forecast history found</h4>
            <p className="text-sm max-w-sm mt-1">
              Forecasts generated from your sales data will be automatically saved here for historical tracking.
            </p>
          </div>
        ) : (
          <div className="border rounded-lg overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Horizon</TableHead>
                  <TableHead>Saved Date</TableHead>
                  <TableHead>Forecast Window</TableHead>
                  <TableHead>Historical Context</TableHead>
                  <TableHead className="text-right">Projected Total</TableHead>
                  <TableHead className="text-right">Daily Average</TableHead>
                  <TableHead className="text-right">Growth</TableHead>
                  <TableHead className="text-center w-[120px]">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {history.map((item) => (
                  <TableRow key={item.id} className="hover:bg-muted/40 transition-colors">
                    <TableCell>
                      <Badge variant="secondary" className="font-semibold">
                        {formatHorizon(item.forecast_days)}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-xs text-muted-foreground whitespace-nowrap">
                      {formatDateTime(item.created_at)}
                    </TableCell>
                    <TableCell className="text-xs font-mono whitespace-nowrap">
                      {(() => {
                        const win = getForecastDateWindow(item);
                        return <>{win.start} <span className="text-muted-foreground">→</span> {win.end}</>;
                      })()}
                    </TableCell>
                    <TableCell className="text-xs font-mono text-muted-foreground whitespace-nowrap">
                      {item.historical_start_date && item.historical_end_date ? (
                        `${item.historical_start_date} → ${item.historical_end_date}`
                      ) : (
                        "—"
                      )}
                    </TableCell>
                    <TableCell className="text-right font-semibold tabular-nums whitespace-nowrap">
                      {formatCurrency(item.forecast_total)}
                    </TableCell>
                    <TableCell className="text-right text-muted-foreground tabular-nums whitespace-nowrap">
                      {formatCurrency(item.forecast_average)}
                    </TableCell>
                    <TableCell className="text-right tabular-nums whitespace-nowrap">
                      <span className={cn(
                        "font-semibold",
                        (item.expected_growth_percentage ?? 0) >= 0 ? "text-green-600 dark:text-green-400" : "text-destructive"
                      )}>
                        {(item.expected_growth_percentage ?? 0) >= 0 ? "+" : ""}
                        {item.expected_growth_percentage}%
                      </span>
                    </TableCell>
                    <TableCell className="text-center">
                      <div className="flex items-center justify-center gap-1">
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-8 w-8 hover:text-primary"
                          title="View Forecast Details"
                          onClick={() => handleViewDetail(item.id)}
                        >
                          <Eye className="size-4" />
                          <span className="sr-only">View</span>
                        </Button>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-8 w-8 text-destructive hover:bg-destructive/10"
                          title="Delete Forecast"
                          onClick={() => handleOpenDelete(item)}
                        >
                          <Trash2 className="size-4" />
                          <span className="sr-only">Delete</span>
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        )}
      </Panel>

      {/* Add / Edit Daily Sales Dialog */}
      <Dialog open={dailySalesDialogOpen} onOpenChange={setDailySalesDialogOpen}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Calendar className="size-5 text-primary" />
              <span>{editingEntry ? "Edit Daily Sales Entry" : "Record Daily Sales"}</span>
            </DialogTitle>
            <DialogDescription>
              {editingEntry 
                ? "Update revenue for this specific calendar date." 
                : "Enter your daily business sales revenue in INR."}
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4 py-2">
            <div className="space-y-1.5">
              <Label htmlFor="daily-sales-date">Sales Date</Label>
              <Input
                id="daily-sales-date"
                type="date"
                value={inputDate}
                onChange={(e) => setInputDate(e.target.value)}
                disabled={Boolean(editingEntry) || isSavingDailySales}
                max={new Date().toISOString().split("T")[0]}
              />
              {editingEntry ? (
                <p className="text-xs text-muted-foreground">The date cannot be changed when editing an existing entry.</p>
              ) : null}
            </div>

            <div className="space-y-1.5">
              <Label htmlFor="daily-sales-revenue">Daily Revenue (₹ INR)</Label>
              <div className="relative">
                <span className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground font-semibold text-sm">₹</span>
                <Input
                  id="daily-sales-revenue"
                  type="number"
                  step="0.01"
                  min="0"
                  placeholder="0.00"
                  className="pl-7"
                  value={inputRevenue}
                  onChange={(e) => setInputRevenue(e.target.value)}
                  disabled={isSavingDailySales}
                />
              </div>
            </div>

            {formError ? (
              <div className="p-3 rounded-md bg-destructive/10 border border-destructive/20 text-destructive text-sm flex items-start gap-2">
                <AlertCircle className="size-4 shrink-0 mt-0.5" />
                <span>{formError}</span>
              </div>
            ) : null}
          </div>

          <DialogFooter className="gap-2 sm:gap-0">
            <Button 
              type="button" 
              variant="outline" 
              onClick={() => setDailySalesDialogOpen(false)}
              disabled={isSavingDailySales}
            >
              Cancel
            </Button>
            <Button 
              type="button" 
              onClick={handleSaveDailySales}
              disabled={isSavingDailySales}
            >
              {isSavingDailySales ? (
                <>
                  <Loader2 className="size-4 animate-spin mr-2" />
                  Saving...
                </>
              ) : (
                editingEntry ? "Update Sales" : "Save Sales"
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Forecast Detail Modal */}
      <Dialog open={detailModalOpen} onOpenChange={setDetailModalOpen}>
        <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-xl">
              <LineChartIcon className="size-5 text-primary" />
              <span>
                {detailRecord ? `Forecast Details — ${formatHorizon(detailRecord.forecast_days)}` : "Forecast Details"}
              </span>
            </DialogTitle>
            <DialogDescription>
              {detailRecord ? (
                <>Generated and saved on {formatDateTime(detailRecord.created_at)}</>
              ) : (
                <>Loading forecast information...</>
              )}
            </DialogDescription>
          </DialogHeader>

          {detailLoading ? (
            <div className="flex flex-col items-center justify-center py-16 gap-3">
              <Loader2 className="size-8 animate-spin text-primary" />
              <p className="text-sm text-muted-foreground">Retrieving full forecast metrics and chart records...</p>
            </div>
          ) : detailError ? (
            <div className="p-4 rounded-lg bg-destructive/10 border border-destructive/20 text-destructive flex items-start gap-3 my-4">
              <AlertCircle className="size-5 shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold">Unable to load forecast</h4>
                <p className="text-sm">{detailError}</p>
              </div>
            </div>
          ) : detailRecord ? (
            <div className="space-y-6 pt-2">
              {/* Summary KPIs */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="p-3 bg-muted/40 rounded-lg border space-y-1">
                  <span className="text-xs text-muted-foreground block">Historical Period</span>
                  <p className="text-sm font-semibold truncate">
                    {detailRecord.historical_start_date && detailRecord.historical_end_date 
                      ? `${detailRecord.historical_start_date} → ${detailRecord.historical_end_date}` 
                      : "—"}
                  </p>
                  <p className="text-xs text-muted-foreground">
                    Avg: {formatCurrency(detailRecord.historical_average_daily_sales)} / day
                  </p>
                </div>
                <div className="p-3 bg-muted/40 rounded-lg border space-y-1">
                  <span className="text-xs text-muted-foreground block">Forecast Horizon</span>
                  <p className="text-sm font-semibold">{formatHorizon(detailRecord.forecast_days)}</p>
                  <p className="text-xs text-muted-foreground truncate">
                    {(() => {
                      const win = getForecastDateWindow(detailRecord);
                      return `${win.start} → ${win.end}`;
                    })()}
                  </p>
                </div>
                <div className="p-3 bg-muted/40 rounded-lg border space-y-1">
                  <span className="text-xs text-muted-foreground block">Forecast Total</span>
                  <p className="text-lg font-bold text-primary">{formatCurrency(detailRecord.forecast_total)}</p>
                  <p className="text-xs text-muted-foreground">
                    {formatCurrency(detailRecord.forecast_average)} / day avg
                  </p>
                </div>
                <div className="p-3 bg-muted/40 rounded-lg border space-y-1">
                  <span className="text-xs text-muted-foreground block">Expected Growth</span>
                  <p className={cn(
                    "text-lg font-bold",
                    (detailRecord.expected_growth_percentage ?? 0) >= 0 ? "text-green-600 dark:text-green-400" : "text-destructive"
                  )}>
                    {(detailRecord.expected_growth_percentage ?? 0) >= 0 ? "+" : ""}
                    {detailRecord.expected_growth_percentage}%
                  </p>
                  <p className="text-xs text-muted-foreground">Seasonal forecast trajectory</p>
                </div>
              </div>

              {/* Chart */}
              {detailRecord.forecast_records && detailRecord.forecast_records.length > 0 && (
                <div className="space-y-2">
                  <h4 className="text-sm font-semibold">Forecast Trajectory & Confidence Bounds</h4>
                  <div className="h-[320px] w-full border rounded-lg p-2 bg-background/50">
                    <ResponsiveContainer width="100%" height="100%">
                      <ComposedChart data={detailRecord.forecast_records} margin={{ top: 10, right: 20, bottom: 10, left: 10 }}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--color-border)" />
                        <XAxis 
                          dataKey="date" 
                          stroke="var(--color-muted-foreground)"
                          fontSize={11}
                          tickMargin={8}
                        />
                        <YAxis 
                          stroke="var(--color-muted-foreground)"
                          fontSize={11}
                          tickFormatter={(value) => `₹${value / 1000}k`}
                        />
                        <Tooltip 
                          formatter={(value: number) => formatCurrency(value)}
                          contentStyle={{ borderRadius: 8, border: "1px solid var(--color-border)" }}
                        />
                        <Legend />
                        <Area 
                          type="monotone" 
                          dataKey={["lower_bound", "upper_bound"] as any}
                          stroke="none" 
                          fill="var(--color-primary)" 
                          fillOpacity={0.15} 
                          name="95% Confidence Interval"
                        />
                        <Line 
                          type="monotone" 
                          dataKey="predicted_sales" 
                          stroke="var(--color-primary)" 
                          strokeWidth={2}
                          dot={false}
                          name="Forecast Sales"
                        />
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}

              {/* Detailed Predictions Table */}
              {detailRecord.forecast_records && detailRecord.forecast_records.length > 0 && (
                <div className="space-y-2">
                  <h4 className="text-sm font-semibold">Daily Projections ({detailRecord.forecast_records.length} days)</h4>
                  <div className="max-h-[260px] overflow-auto border rounded-lg">
                    <Table>
                      <TableHeader className="sticky top-0 bg-background z-10">
                        <TableRow>
                          <TableHead>Date</TableHead>
                          <TableHead className="text-right">Projected Sales</TableHead>
                          <TableHead className="text-right">Lower Bound (95%)</TableHead>
                          <TableHead className="text-right">Upper Bound (95%)</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {detailRecord.forecast_records.map((r, i) => (
                          <TableRow key={i}>
                            <TableCell className="font-medium text-xs">{r.date}</TableCell>
                            <TableCell className="text-right font-bold text-primary tabular-nums text-xs">
                              {formatCurrency(r.predicted_sales)}
                            </TableCell>
                            <TableCell className="text-right text-muted-foreground tabular-nums text-xs">
                              {formatCurrency(r.lower_bound)}
                            </TableCell>
                            <TableCell className="text-right text-muted-foreground tabular-nums text-xs">
                              {formatCurrency(r.upper_bound)}
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                </div>
              )}
            </div>
          ) : null}
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation Alert Dialog */}
      <AlertDialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Forecast Record?</AlertDialogTitle>
            <AlertDialogDescription>
              This will permanently delete the {recordToDelete ? formatHorizon(recordToDelete.forecast_days) : ""} forecast
              record generated on {recordToDelete ? formatDateTime(recordToDelete.created_at) : ""}. This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={isDeleting}>Cancel</AlertDialogCancel>
            <AlertDialogAction 
              onClick={(e) => {
                e.preventDefault();
                handleConfirmDelete();
              }}
              disabled={isDeleting}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              {isDeleting ? (
                <>
                  <Loader2 className="size-4 animate-spin mr-2" />
                  Deleting...
                </>
              ) : (
                "Delete Forecast"
              )}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
