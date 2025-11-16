import { useState, useEffect, useRef, useCallback } from "react";
import "@/App.css";
import axios from "axios";
import { Search, Upload, FileText, Image as ImageIcon, File, Filter, Download, Tag, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { toast } from "sonner";
import { Toaster } from "@/components/ui/sonner";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function App() {
  const [searchQuery, setSearchQuery] = useState("");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [showUpload, setShowUpload] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [stats, setStats] = useState({ total_documents: 0, tags: [] });
  const [selectedFileType, setSelectedFileType] = useState("all");
  const [selectedTags, setSelectedTags] = useState([]);
  const [showFilters, setShowFilters] = useState(false);
  const searchTimeoutRef = useRef(null);
  const searchCacheRef = useRef(new Map());

  useEffect(() => {
    fetchStats();
    performSearch("", "all", []);
  }, []);

  const fetchStats = async () => {
    try {
      const response = await axios.get(`${API}/stats`);
      setStats(response.data);
    } catch (error) {
      console.error("Error fetching stats:", error);
    }
  };

  const performSearch = useCallback(async (query = searchQuery, fileType = selectedFileType, tags = selectedTags) => {
    setLoading(true);
    
    const cacheKey = `${query}|${fileType}|${tags.sort().join(',')}`;
    
    if (searchCacheRef.current.has(cacheKey)) {
      const cachedResults = searchCacheRef.current.get(cacheKey);
      setResults(cachedResults);
      setLoading(false);
      return;
    }
    
    try {
      let fileTypes = null;
      if (fileType !== "all") {
        if (fileType === "image/") {
          fileTypes = ["image/"];
        } else {
          fileTypes = [fileType];
        }
      }
      
      const response = await axios.post(`${API}/documents/search`, {
        query: query,
        file_types: fileTypes,
        tags: tags.length > 0 ? tags : null,
        limit: 50
      }, {
        timeout: 10000
      });
      
      const resultsData = response.data.results || [];
      
      searchCacheRef.current.set(cacheKey, resultsData);
      
      if (searchCacheRef.current.size > 100) {
        const firstKey = searchCacheRef.current.keys().next().value;
        searchCacheRef.current.delete(firstKey);
      }
      
      setResults(resultsData);
    } catch (error) {
      console.error("Error searching:", error);
      if (error.code === 'ECONNABORTED') {
        toast.error("Search request timed out. Please try again.");
      } else {
        toast.error("Failed to search documents. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  }, [searchQuery, selectedFileType, selectedTags]);

  const handleSearch = (e) => {
    e.preventDefault();
    performSearch(searchQuery, selectedFileType, selectedTags);
  };

  const handleSearchInputChange = (e) => {
    const value = e.target.value;
    setSearchQuery(value);
    
    if (searchTimeoutRef.current) {
      clearTimeout(searchTimeoutRef.current);
    }
    
    searchTimeoutRef.current = setTimeout(() => {
      performSearch(value, selectedFileType, selectedTags);
    }, 300);
  };

  const handleImmediateSearch = () => {
    if (searchTimeoutRef.current) {
      clearTimeout(searchTimeoutRef.current);
    }
    performSearch(searchQuery, selectedFileType, selectedTags);
  };

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      if (file.size > 20 * 1024 * 1024) {
        toast.error("File size must be less than 20MB");
        return;
      }
      setSelectedFile(file);
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.size > 20 * 1024 * 1024) {
        toast.error("File size must be less than 20MB");
        return;
      }
      setSelectedFile(file);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setUploading(true);
    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await axios.post(`${API}/documents/upload`, formData, {
        headers: { "Content-Type": "multipart/form-data" }
      });
      toast.success(`${selectedFile.name} uploaded successfully!`);
      setSelectedFile(null);
      setShowUpload(false);
      fetchStats();
      performSearch(searchQuery, selectedFileType, selectedTags);
      clearSearchCache();
    } catch (error) {
      console.error("Error uploading:", error);
      toast.error("Failed to upload document");
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async (docId, docName) => {
    if (!window.confirm(`Delete "${docName}"?`)) return;

    try {
      await axios.delete(`${API}/documents/${docId}`);
      toast.success("Document deleted");
      fetchStats();
      setResults(prevResults => prevResults.filter(doc => doc.id !== docId));
      clearSearchCache();
    } catch (error) {
      console.error("Error deleting:", error);
      toast.error("Failed to delete document");
    }
  };

  const handleDownload = async (docId, docName) => {
    try {
      const response = await axios.get(`${API}/documents/${docId}/download`, {
        responseType: "blob"
      });
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", docName);
      document.body.appendChild(link);
      link.click();
      link.remove();
      toast.success("Download started");
    } catch (error) {
      console.error("Error downloading:", error);
      toast.error("Failed to download document");
    }
  };

  const getFileIcon = (fileType) => {
    if (fileType.includes("pdf")) return <FileText className="w-8 h-8" />;
    if (fileType.includes("image")) return <ImageIcon className="w-8 h-8" />;
    return <File className="w-8 h-8" />;
  };

  const formatFileSize = (bytes) => {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
  };

  const formatDate = (dateStr) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString() + " " + date.toLocaleTimeString();
  };

  const toggleTag = (tag) => {
    if (selectedTags.includes(tag)) {
      const newTags = selectedTags.filter(t => t !== tag);
      setSelectedTags(newTags);
      performSearch(searchQuery, selectedFileType, newTags);
    } else {
      const newTags = [...selectedTags, tag];
      setSelectedTags(newTags);
      performSearch(searchQuery, selectedFileType, newTags);
    }
    clearSearchCache();
  };

  const handleFileTypeChange = (fileType) => {
    setSelectedFileType(fileType);
    performSearch(searchQuery, fileType, selectedTags);
    clearSearchCache();
  };

  const clearFilters = () => {
    setSelectedFileType("all");
    setSelectedTags([]);
    setSearchQuery("");
    performSearch("", "all", []);
  };

  const clearSearchCache = () => {
    searchCacheRef.current.clear();
  };

  return (
    <div className="App min-h-screen">
      <Toaster position="top-right" />
      
      <header className="header-gradient border-b border-white/10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl sm:text-4xl font-bold text-white mb-2">Findify</h1>
              <p className="text-white/80">Search and discover your documents instantly</p>
            </div>
            <Button 
              onClick={() => setShowUpload(true)} 
              className="btn-primary"
              data-testid="upload-btn"
            >
              <Upload className="w-4 h-4 mr-2" />
              Upload
            </Button>
          </div>
          
          <form onSubmit={handleSearch} className="mt-8">
            <div className="search-bar">
              <Search className="w-5 h-5 text-gray-400" />
              <Input
                type="text"
                placeholder="Search documents, keywords, or tags..."
                value={searchQuery}
                onChange={handleSearchInputChange}
                className="search-input"
                data-testid="search-input"
              />
              <Button 
                type="button"
                variant="ghost" 
                size="sm"
                className="search-btn"
                onClick={handleImmediateSearch}
              >
                <Search className="w-4 h-4" />
              </Button>
              <Button 
                type="button" 
                variant="ghost" 
                size="sm"
                onClick={() => setShowFilters(!showFilters)}
                className="filter-btn"
                data-testid="filter-btn"
              >
                <Filter className="w-4 h-4" />
              </Button>
            </div>
          </form>
          
          <div className="flex items-center gap-6 mt-4 text-sm text-white/70">
            <span>{stats.total_documents} documents</span>
            {(selectedFileType !== "all" || selectedTags.length > 0) && (
              <button onClick={clearFilters} className="text-blue-300 hover:text-blue-200">
                Clear filters
              </button>
            )}
            {loading && (
              <div className="flex items-center text-blue-200">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-200 mr-2"></div>
                <span>Searching...</span>
              </div>
            )}
          </div>
        </div>
      </header>

      {showFilters && (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 border-b border-gray-200">
          <div className="flex flex-wrap gap-4">
            <div>
              <label className="text-sm font-medium text-gray-700 mb-2 block">File Type</label>
              <Select value={selectedFileType} onValueChange={handleFileTypeChange}>
                <SelectTrigger className="w-[180px]" data-testid="file-type-select">
                  <SelectValue placeholder="All types" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All types</SelectItem>
                  <SelectItem value="application/pdf">PDF</SelectItem>
                  <SelectItem value="application/vnd.openxmlformats-officedocument.wordprocessingml.document">Word</SelectItem>
                  <SelectItem value="application/vnd.openxmlformats-officedocument.presentationml.presentation">PowerPoint</SelectItem>
                  <SelectItem value="text/plain">Text</SelectItem>
                  <SelectItem value="image/">Images</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {stats.tags && stats.tags.length > 0 && (
              <div className="flex-1">
                <label className="text-sm font-medium text-gray-700 mb-2 block">Tags</label>
                <div className="flex flex-wrap gap-2">
                  {stats.tags.map(tag => (
                    <Badge
                      key={tag}
                      variant={selectedTags.includes(tag) ? "default" : "outline"}
                      className="cursor-pointer"
                      onClick={() => toggleTag(tag)}
                      data-testid={`tag-${tag}`}
                    >
                      {tag}
                    </Badge>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {loading ? (
          <div className="text-center py-12">
            <div className="inline-block animate-spin rounded-full h-12 w-12 border-4 border-gray-300 border-t-blue-500"></div>
            <p className="mt-4 text-gray-600">Searching...</p>
          </div>
        ) : results.length === 0 ? (
          <div className="text-center py-12">
            <File className="w-16 h-16 text-gray-300 mx-auto mb-4" />
            <p className="text-gray-500 text-lg">No documents found</p>
            <Button 
              onClick={() => setShowUpload(true)} 
              variant="outline" 
              className="mt-4"
              data-testid="empty-upload-btn"
            >
              <Upload className="w-4 h-4 mr-2" />
              Upload your first document
            </Button>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {results.map((doc) => (
              <Card key={doc.id} className="result-card" data-testid={`doc-card-${doc.id}`}>
                <CardHeader>
                  <div className="flex items-start justify-between">
                    <div className="flex items-center gap-3">
                      <div className="text-blue-500">
                        {getFileIcon(doc.file_type)}
                      </div>
                      <div className="flex-1">
                        <CardTitle className="text-base line-clamp-1">{doc.name}</CardTitle>
                        <CardDescription className="text-xs mt-1">
                          {formatFileSize(doc.size)} • {formatDate(doc.created_at)}
                        </CardDescription>
                      </div>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  {doc.snippet && (
                    <p className="text-sm text-gray-600 line-clamp-3 mb-3">{doc.snippet}</p>
                  )}
                  {doc.tags && doc.tags.length > 0 && (
                    <div className="flex flex-wrap gap-1">
                      {doc.tags.map((tag, idx) => (
                        <Badge key={idx} variant="secondary" className="text-xs">
                          {tag}
                        </Badge>
                      ))}
                    </div>
                  )}
                </CardContent>
                <CardFooter className="flex gap-2">
                  <Button 
                    size="sm" 
                    variant="outline" 
                    className="flex-1"
                    onClick={() => handleDownload(doc.id, doc.name)}
                    data-testid={`download-btn-${doc.id}`}
                  >
                    <Download className="w-4 h-4 mr-1" />
                    Download
                  </Button>
                  <Button 
                    size="sm" 
                    variant="ghost" 
                    onClick={() => handleDelete(doc.id, doc.name)}
                    data-testid={`delete-btn-${doc.id}`}
                  >
                    <Trash2 className="w-4 h-4" />
                  </Button>
                </CardFooter>
              </Card>
            ))}
          </div>
        )}
      </main>

      <Dialog open={showUpload} onOpenChange={setShowUpload}>
        <DialogContent className="sm:max-w-md" data-testid="upload-dialog">
          <DialogHeader>
            <DialogTitle>Upload Document</DialogTitle>
            <DialogDescription>
              Upload PDFs, Word docs, PowerPoint, images, or text files (max 20MB)
            </DialogDescription>
          </DialogHeader>
          <div
            className={`upload-zone ${dragActive ? 'drag-active' : ''}`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            data-testid="upload-zone"
          >
            <input
              type="file"
              onChange={handleFileSelect}
              accept=".pdf,.doc,.docx,.ppt,.pptx,.txt,.png,.jpg,.jpeg"
              className="hidden"
              id="file-upload"
              data-testid="file-input"
            />
            <label htmlFor="file-upload" className="cursor-pointer text-center">
              <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
              <p className="text-sm text-gray-600 mb-2">
                {selectedFile ? selectedFile.name : "Drag and drop or click to select"}
              </p>
              <p className="text-xs text-gray-400">
                PDF, DOCX, PPTX, TXT, PNG, JPG • Max 20MB
              </p>
            </label>
          </div>
          <div className="flex gap-2">
            <Button 
              onClick={handleUpload} 
              disabled={!selectedFile || uploading} 
              className="flex-1"
              data-testid="upload-submit-btn"
            >
              {uploading ? "Uploading..." : "Upload"}
            </Button>
            <Button 
              onClick={() => {
                setShowUpload(false);
                setSelectedFile(null);
              }} 
              variant="outline"
              data-testid="upload-cancel-btn"
            >
              Cancel
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}

export default App;