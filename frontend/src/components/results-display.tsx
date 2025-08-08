"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { Eye, Activity, TrendingUp, Info } from "lucide-react";

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
      {/* Imagen procesada */}
      <Card>
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

      {/* Métricas principales */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Tortuosidad Promedio</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{data.avg_tortuosity.toFixed(3)}</div>
            <p className="text-xs text-muted-foreground">
              Valor global de tortuosidad
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Glándulas Detectadas</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{data.num_glands}</div>
            <p className="text-xs text-muted-foreground">
              Total de glándulas identificadas
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Rango de Valores</CardTitle>
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
      </div>

      {/* Tabla de tortuosidad individual */}
      <Card>
        <CardHeader>
          <CardTitle>Tortuosidad por Glándula Individual</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>ID Glándula</TableHead>
                <TableHead>Valor de Tortuosidad</TableHead>
                <TableHead>Interpretación</TableHead>
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
        </CardContent>
      </Card>

      {/* Gráfico de barras */}
      <Card>
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

      {/* Información detallada */}
      <Card>
        <CardHeader>
          <CardTitle>Información Detallada de Tortuosidad</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <h4 className="font-semibold mb-2">¿Qué es la Tortuosidad?</h4>
            <p className="text-sm text-muted-foreground">
              La tortuosidad es una medida de cuán retorcida o curvada está una glándula de Meibomio. 
              Un valor más alto indica una glándula más tortuosa, lo que puede ser un indicador de 
              disfunción de las glándulas de Meibomio (MGD).
            </p>
          </div>
          
          <div>
            <h4 className="font-semibold mb-2">Interpretación de los valores (referencial):</h4>
            <ul className="space-y-1 text-sm text-muted-foreground">
              <li><strong>0.0 - 0.1:</strong> Tortuosidad baja (generalmente normal)</li>
              <li><strong>0.1 - 0.2:</strong> Tortuosidad moderada (puede indicar cambios iniciales)</li>
              <li><strong>&gt; 0.2:</strong> Tortuosidad alta (sugestivo de MGD, requiere correlación clínica)</li>
            </ul>
          </div>
          
          <p className="text-xs text-muted-foreground italic">
            Nota: Estos rangos son aproximados y la interpretación final debe ser realizada por un especialista.
          </p>
        </CardContent>
      </Card>
    </div>
  );
} 