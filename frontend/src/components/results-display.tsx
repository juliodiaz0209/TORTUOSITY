"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { Eye, Activity, TrendingUp, Info, BarChart3 } from "lucide-react";

interface TortuosityData {
  avg_tortuosity: number;
  num_glands: number;
  individual_tortuosities: number[];
  analysis_info: {
    total_glands_analyzed: number;
    tortuosity_range: {
      min: number;
      max: number;
    };
  };
}

interface ResultsDisplayProps {
  data: TortuosityData;
  processedImage: string;
}

export function ResultsDisplay({ data, processedImage }: ResultsDisplayProps) {
  const getInterpretation = (value: number) => {
    if (value <= 0.1) return { text: "Baja (Normal)", color: "bg-green-500" };
    if (value <= 0.2) return { text: "Moderada", color: "bg-yellow-500" };
    return { text: "Alta (MGD)", color: "bg-red-500" };
  };

  const chartData = data.individual_tortuosities.map((value, index) => ({
    gland: `G${index + 1}`,
    tortuosity: value,
    interpretation: getInterpretation(value).text,
  }));

  return (
    <div className="space-y-6">
      {/* Main Metrics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="border-border">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Tortuosidad Promedio</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{data.avg_tortuosity.toFixed(3)}</div>
            <p className="text-xs text-muted-foreground">
              Valor global
            </p>
          </CardContent>
        </Card>

        <Card className="border-border">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Glándulas Detectadas</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{data.num_glands}</div>
            <p className="text-xs text-muted-foreground">
              Total identificadas
            </p>
          </CardContent>
        </Card>

        <Card className="border-border">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Rango Mín-Máx</CardTitle>
            <Info className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {data.analysis_info.tortuosity_range.min.toFixed(3)} - {data.analysis_info.tortuosity_range.max.toFixed(3)}
            </div>
            <p className="text-xs text-muted-foreground">
              Mínimo - Máximo
            </p>
          </CardContent>
        </Card>

        <Card className="border-border">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Estado General</CardTitle>
            <BarChart3 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {data.avg_tortuosity <= 0.1 ? "Normal" : data.avg_tortuosity <= 0.2 ? "Moderado" : "Alto"}
            </div>
            <p className="text-xs text-muted-foreground">
              {getInterpretation(data.avg_tortuosity).text}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Image and Table Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Processed Image */}
        <Card className="border-border">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Eye className="h-5 w-5" />
              Imagen Procesada
            </CardTitle>
          </CardHeader>
          <CardContent>
            <img
              src={processedImage}
              alt="Imagen procesada"
              className="w-full rounded-lg shadow-lg"
            />
          </CardContent>
        </Card>

        {/* Individual Tortuosity Table */}
        <Card className="border-border">
          <CardHeader>
            <CardTitle>Tortuosidad por Glándula</CardTitle>
          </CardHeader>
          <CardContent>
                         <div className="max-h-96 overflow-y-auto custom-scrollbar">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>ID</TableHead>
                    <TableHead>Valor</TableHead>
                    <TableHead>Estado</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {data.individual_tortuosities.map((value, index) => {
                    const interpretation = getInterpretation(value);
                    return (
                      <TableRow key={index}>
                        <TableCell className="font-medium">G{index + 1}</TableCell>
                        <TableCell>{value.toFixed(3)}</TableCell>
                        <TableCell>
                          <Badge className={interpretation.color}>
                            {interpretation.text}
                          </Badge>
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Chart */}
      <Card className="border-border">
        <CardHeader>
          <CardTitle>Visualización Gráfica de Tortuosidad</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-[400px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="gland" />
                <YAxis />
                <Tooltip
                  formatter={(value: number) => [value.toFixed(3), "Tortuosidad"]}
                  labelFormatter={(label) => `Glándula ${label}`}
                />
                <Bar
                  dataKey="tortuosity"
                  fill="#3b82f6"
                  radius={[4, 4, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="text-center text-sm text-muted-foreground mt-4">
            Gráfico comparativo de la tortuosidad para cada glándula identificada. 
            Valores más altos indican mayor curvatura.
          </p>
        </CardContent>
      </Card>

      {/* Interpretation Guide */}
      <Card className="border-border">
        <CardHeader>
          <CardTitle>Guía de Interpretación</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="text-center p-4 bg-green-50 dark:bg-green-950 rounded-lg">
              <div className="text-2xl font-bold text-green-600">0.0 - 0.1</div>
              <div className="text-sm text-green-700 dark:text-green-300">Tortuosidad Baja</div>
              <div className="text-xs text-green-600 dark:text-green-400">Normal</div>
            </div>
            <div className="text-center p-4 bg-yellow-50 dark:bg-yellow-950 rounded-lg">
              <div className="text-2xl font-bold text-yellow-600">0.1 - 0.2</div>
              <div className="text-sm text-yellow-700 dark:text-yellow-300">Tortuosidad Moderada</div>
              <div className="text-xs text-yellow-600 dark:text-yellow-400">Cambios Iniciales</div>
            </div>
            <div className="text-center p-4 bg-red-50 dark:bg-red-950 rounded-lg">
              <div className="text-2xl font-bold text-red-600">&gt; 0.2</div>
              <div className="text-sm text-red-700 dark:text-red-300">Tortuosidad Alta</div>
              <div className="text-xs text-red-600 dark:text-red-400">Sugestivo de MGD</div>
            </div>
          </div>
          <p className="text-xs text-muted-foreground italic text-center">
            Nota: Estos rangos son aproximados y la interpretación final debe ser realizada por un especialista.
          </p>
        </CardContent>
      </Card>
    </div>
  );
} 