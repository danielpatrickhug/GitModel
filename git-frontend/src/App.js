import React, { useState } from "react";
import {
  AppBar,
  Box,
  Button,
  Container,
  Grid,
  Paper,
  TextField,
  Toolbar,
  Typography,
} from "@mui/material";
import { GitHub } from "@mui/icons-material";
import "./App.css";

function App() {
  const [repoUrl, setRepoUrl] = useState("");
  const [openaiSecret, setOpenaiSecret] = useState("");
  const [results, setResults] = useState(null);

  const handleRepoUrlChange = (event) => {
    setRepoUrl(event.target.value);
  };

  const handleOpenaiSecretChange = (event) => {
    setOpenaiSecret(event.target.value);
  };

  const handleSubmit = async () => {
    // Replace this URL with the actual API endpoint of your FastAPI backend
    const apiUrl = "http://localhost:8000/api/topic-modeling";

    try {
      const response = await fetch(apiUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          config: "./test_config.yaml", // Update this as needed
          repo: repoUrl,
          repo_name: "gitmodel", // Update this as needed
          openai_secret: openaiSecret,
        }),
      });

      const data = await response.json();
      setResults(data);
    } catch (error) {
      console.error("Error fetching data:", error);
    }
  };

  return (
    <div className="App">
      <AppBar position="static" className="App-header">
        <Toolbar>
          <GitHub fontSize="large" />
          <Typography
            variant="h6"
            component="div"
            sx={{ flexGrow: 1, marginLeft: 2 }}
          >
            GitHub Repo Topic Modeler
          </Typography>
        </Toolbar>
      </AppBar>

      <Container maxWidth="sm">
        <Box sx={{ marginTop: 4 }}>
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <TextField
                label="GitHub Repository URL"
                variant="outlined"
                fullWidth
                value={repoUrl}
                onChange={handleRepoUrlChange}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                label="OpenAI Secret Key"
                variant="outlined"
                fullWidth
                type="password"
                value={openaiSecret}
                onChange={handleOpenaiSecretChange}
              />
            </Grid>
            <Grid item xs={12}>
              <Button
                variant="contained"
                color="primary"
                onClick={handleSubmit}
                fullWidth
              >
                Analyze Repository
              </Button>
            </Grid>
          </Grid>
        </Box>

        {results && (
          <Paper className="results-container">
            <Typography variant="h5" align="center" gutterBottom>
              Topic Modeling Results
            </Typography>

            {/* Display the results in a desired format */}
            <pre>{JSON.stringify(results, null, 2)}</pre>
          </Paper>
        )}
      </Container>
    </div>
  );
}

export default App;
