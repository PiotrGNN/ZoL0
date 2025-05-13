import React, { useEffect, useState } from 'react';
import { Container, Typography, Box, Paper, Grid, CircularProgress, Alert } from '@mui/material';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface StatusResponse {
  components: Record<string, string>;
  status: string;
  timestamp: string;
}

const fetchStatus = async (): Promise<StatusResponse> => {
  const { data } = await axios.get('/health');
  return data;
};

const App: React.FC = () => {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchStatus()
      .then(setStatus)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  return (
    <Container maxWidth="md" sx={{ mt: 4 }}>
      <Paper elevation={3} sx={{ p: 4 }}>
        <Typography variant="h4" gutterBottom>
          Trading System Dashboard
        </Typography>
        {loading && <CircularProgress />}
        {error && <Alert severity="error">{error}</Alert>}
        {status && (
          <Box>
            <Typography variant="h6">System status: {status.status}</Typography>
            <Typography variant="body2" color="text.secondary">
              Last update: {new Date(status.timestamp).toLocaleString()}
            </Typography>
            <Box mt={2}>
              <Grid container spacing={2}>
                {Object.entries(status.components).map(([name, value]) => (
                  <Grid item xs={12} sm={4} key={name}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="subtitle1">{name}</Typography>
                      <Typography variant="h6" color={value === 'active' ? 'green' : 'gray'}>
                        {value}
                      </Typography>
                    </Paper>
                  </Grid>
                ))}
              </Grid>
            </Box>
          </Box>
        )}
      </Paper>
      <Box mt={4}>
        <Paper elevation={2} sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Example Chart (Random Data)
          </Typography>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={Array.from({ length: 20 }, (_, i) => ({ x: i, y: Math.random() * 100 }))}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="x" />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="y" stroke="#1976d2" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </Paper>
      </Box>
    </Container>
  );
};

export default App;
