import google.generativeai as genai
import json
import time
from typing import Dict, Any, Optional, List  
import os  
import requests    
from dotenv import load_dotenv  
import atexit  
import signal
import sys
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import shutil
from warnings import filterwarnings
filterwarnings("ignore")
load_dotenv()


class SmartSearchDecider:
    """Intelligent decision maker for when to use RAG vs Search vs Both"""
    
    def __init__(self):
        # Topics that are stable and should primarily use RAG
        self.stable_topics = {
            'process': ['buying process', 'selling process', 'closing process', 'mortgage process'],
            'definitions': ['what is', 'explain', 'define', 'how does', 'how do'],
            'historical': ['history', 'established', 'founded', 'originally'],
            'features': ['amenities', 'characteristics', 'features', 'lifestyle'],
            'geography': ['location', 'boundaries', 'distance', 'area']
        }
        
        # Topics that always need current data
        self.dynamic_topics = {
            'market': ['price', 'rates', 'market', 'trends', 'inventory', 'sales'],
            'temporal': ['current', 'latest', 'recent', 'today', 'now', '2024', '2025'],
            'statistics': ['average', 'median', 'statistics', 'data', 'numbers'],
            'availability': ['available', 'for sale', 'listings', 'on market'],
            'news': ['news', 'announced', 'update', 'happening', 'events']
        }
        
        # Topics that might need both
        self.hybrid_topics = {
            'neighborhoods': ['coral gables', 'brickell', 'wynwood', 'aventura', 'coconut grove', 'south beach'],
            'comparison': ['compare', 'versus', 'vs', 'better', 'difference'],
            'investment': ['roi', 'invest', 'rental', 'return', 'yield'],
            'comprehensive': ['tell me about', 'overview', 'guide', 'everything']
        }

    def analyze_query(self, user_input: str, rag_retriever=None) -> dict:
        """
        Intelligently determine information sources needed
        
        Returns:
        {
            'use_rag': bool,
            'use_search': bool,
            'rag_query': str,
            'search_query': str,
            'reasoning': str
        }
        """
        input_lower = user_input.lower()
        
        # Initialize decision
        decision = {
            'use_rag': False,
            'use_search': False,
            'rag_query': user_input,
            'search_query': user_input,
            'reasoning': ''
        }
        
        # Check for stable topics (RAG preferred)
        stable_match = False
        for category, keywords in self.stable_topics.items():
            if any(keyword in input_lower for keyword in keywords):
                stable_match = True
                decision['use_rag'] = True
                decision['reasoning'] += f"RAG: {category} information is stable. "
                break
        
        # Check for dynamic topics (Search needed)
        dynamic_match = False
        for category, keywords in self.dynamic_topics.items():
            if any(keyword in input_lower for keyword in keywords):
                dynamic_match = True
                decision['use_search'] = True
                decision['reasoning'] += f"Search: {category} requires current data. "
                
                # Refine search query for better results
                if category == 'market':
                    decision['search_query'] = user_input + " 2024 real estate Miami"
                elif category == 'availability':
                    decision['search_query'] = user_input + " MLS listings"
                break
        
        # Check for hybrid topics (might need both)
        for category, keywords in self.hybrid_topics.items():
            if any(keyword in input_lower for keyword in keywords):
                if category == 'neighborhoods' and dynamic_match:
                    # Neighborhood + current info = both
                    decision['use_rag'] = True
                    decision['use_search'] = True
                    decision['reasoning'] += "Both: Neighborhood details + current market. "
                elif category == 'comparison':
                    # Comparisons often need both
                    decision['use_rag'] = True
                    if not stable_match:
                        decision['use_search'] = True
                    decision['reasoning'] += "Both: Static features + current differences. "
                elif category == 'investment':
                    # Investment queries need both
                    decision['use_rag'] = True
                    decision['use_search'] = True
                    decision['reasoning'] += "Both: Investment strategies + current yields. "
                elif category == 'comprehensive' and any(neighborhood in input_lower for neighborhood in self.hybrid_topics['neighborhoods']):
                    decision['use_rag'] = True
                    if dynamic_match:
                        decision['use_search'] = True
                    decision['reasoning'] += "Comprehensive neighborhood information requested. "
                break
        
        # Default decision if no clear match
        if not decision['use_rag'] and not decision['use_search']:
            # Try RAG first to see if we have relevant content
            if rag_retriever:
                test_results = rag_retriever.retrieve(user_input, k=1)
                if test_results and test_results[0]['relevance_score'] > 0.7:
                    decision['use_rag'] = True
                    decision['reasoning'] = "RAG: High relevance content found. "
                else:
                    # Default to both for general queries
                    decision['use_rag'] = True
                    decision['reasoning'] = "Default: Checking knowledge base first. "
            else:
                # Default to RAG for general queries
                decision['use_rag'] = True
                decision['reasoning'] = "Default: General query, checking knowledge base. "
        
        return decision

    def combine_contexts(self, rag_context: str, search_context: str, query_type: str) -> str:
        """
        Intelligently combine RAG and Search contexts based on query type
        """
        if not search_context and not rag_context:
            return "No relevant information found."
        
        if not search_context:
            return f"Based on knowledge base:\n{rag_context}"
        
        if not rag_context:
            return f"Current information:\n{search_context}"
        
        # Both contexts available - combine intelligently
        if 'market' in query_type or 'price' in query_type:
            # For market queries, prioritize current data but include historical context
            return f"""Market Analysis:

Historical Context and Fundamentals:
{rag_context}

Current Market Conditions:
{search_context}"""
        
        elif 'neighborhood' in query_type:
            # For neighborhood queries, blend characteristics with current info
            return f"""Comprehensive Information:

Neighborhood Characteristics and Features:
{rag_context}

Recent Updates and Current Market:
{search_context}"""
        
        else:
            # Default combination
            return f"""Comprehensive Information:

From Knowledge Base:
{rag_context}

Current Information:
{search_context}"""


class DocumentProcessor:
    """Process and store documents for RAG"""
    
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
    def load_documents(self, directory_path):
        """Load all documents from a directory"""
        documents = []
        
        if not os.path.exists(directory_path):
            print(f"Directory {directory_path} not found. Creating sample documents...")
            self._create_sample_documents(directory_path)
        
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                
                try:
                    if file.endswith('.pdf'):
                        loader = PyPDFLoader(file_path)
                    elif file.endswith('.txt'):
                        loader = TextLoader(file_path, encoding='utf-8')
                    elif file.endswith('.docx'):
                        loader = UnstructuredWordDocumentLoader(file_path)
                    else:
                        continue
                    
                    docs = loader.load()
                    # Add source metadata
                    for doc in docs:
                        doc.metadata['source'] = file
                        doc.metadata['category'] = os.path.basename(root)
                    
                    documents.extend(docs)
                    print(f"Loaded: {file}")
                except Exception as e:
                    print(f"Error loading {file}: {e}")
        
        return documents
    
    def _create_sample_documents(self, directory_path):
        """Create sample knowledge base documents"""
        os.makedirs(directory_path, exist_ok=True)
        
        # Create sample documents
        sample_docs = {
            "general_knowledge.txt": """REAL ESTATE FUNDAMENTALS

HOME BUYING PROCESS
The home buying process involves several crucial steps that every buyer should understand:

1. Financial Preparation
- Get pre-approved for a mortgage
- Determine your budget (typically 28% of gross income for housing)
- Save for down payment (3-20% of purchase price)
- Budget for closing costs (2-5% of purchase price)

2. Finding the Right Property
- Work with a licensed real estate agent
- Define your needs vs wants
- Research neighborhoods thoroughly
- Consider future resale value

3. Making an Offer
- Comparative market analysis (CMA)
- Negotiation strategies
- Contingencies (inspection, financing, appraisal)
- Earnest money deposit

4. Due Diligence Period
- Professional home inspection
- Review HOA documents if applicable
- Verify property boundaries
- Check for liens or encumbrances

5. Closing Process
- Final walkthrough
- Review closing disclosure
- Wire transfer of funds
- Recording of deed

PROPERTY TYPES

Single-Family Homes
- Complete ownership of structure and land
- Privacy and space
- Maintenance responsibility
- Higher cost but better appreciation

Condominiums
- Own the unit interior
- Shared common areas
- HOA fees cover maintenance
- Good for first-time buyers

Townhouses
- Multi-level attached homes
- Small private yards
- Lower maintenance than single-family
- Good compromise option

MORTGAGE BASICS

Types of Mortgages:
- Conventional (20% down avoids PMI)
- FHA (3.5% down, easier qualification)
- VA (0% down for veterans)
- USDA (rural properties, 0% down)

Interest Rates:
- Fixed-rate: Same rate for entire term
- Adjustable-rate (ARM): Rate changes periodically
- Points: Prepaid interest to lower rate

REAL ESTATE TERMINOLOGY

- MLS: Multiple Listing Service
- CMA: Comparative Market Analysis
- HOA: Homeowners Association
- PMI: Private Mortgage Insurance
- LTV: Loan-to-Value Ratio
- DTI: Debt-to-Income Ratio
- Escrow: Third-party holding of funds
- Title Insurance: Protects against ownership disputes""",

            "miami_neighborhoods.txt": """MIAMI-DADE COUNTY NEIGHBORHOOD GUIDE

CORAL GABLES - "The City Beautiful"
Established: 1925
Character: Mediterranean Revival architecture, tree-lined streets
Population: 51,000
Median Home Price: $1.2 million
Price per Sq Ft: $450-$650

Demographics:
- Family-oriented community
- High-income professionals
- International residents (30%)

Schools (All A-rated):
- Coral Gables Senior High (9/10)
- George Washington Carver Middle (8/10)
- Coral Gables Elementary (9/10)

Amenities:
- Miracle Mile shopping district
- Venetian Pool
- Biltmore Golf Course
- University of Miami campus

Best For: Families seeking top schools and historic charm

BRICKELL - "Manhattan of the South"
Character: Urban high-rise living, financial district
Population: 40,000
Median Condo Price: $550,000
Price per Sq Ft: $500-$700

Demographics:
- Young professionals (median age 35)
- International buyers (45%)
- High walkability score

Lifestyle:
- Brickell City Centre (luxury shopping)
- Mary Brickell Village (dining/nightlife)
- Waterfront parks
- Metro access

Best For: Urban professionals, investors

COCONUT GROVE - "The Grove"
Established: 1873 (Miami's oldest neighborhood)
Character: Bohemian, lush canopy, waterfront
Population: 25,000
Median Home Price: $950,000

Features:
- Waterfront dining at Bayshore Drive
- CocoWalk shopping
- Sailing and marina culture
- Arts and culture scene
- Peacock Park

Schools:
- Ransom Everglades (private, elite)
- Coconut Grove Elementary

Best For: Artists, boaters, nature lovers

AVENTURA
Character: Planned community, high-rise condos
Population: 40,000
Median Condo Price: $450,000

Highlights:
- Aventura Mall (largest in Florida)
- Turnberry Golf Course
- Gated communities
- Beach access (3 miles)

Demographics:
- Retirees and seasonal residents
- Family-friendly
- International community

WYNWOOD
Character: Arts district, trendy, gentrifying
Population: 15,000
Median Price: $480,000

Features:
- Wynwood Walls (street art)
- Galleries and studios
- Craft breweries
- Hip restaurants
- Monthly art walks

Best For: Young professionals, artists, investors

SOUTH BEACH (Miami Beach)
Character: Art Deco, beachfront, tourist hub
Median Condo Price: $600,000
Price per Sq Ft: $650-$850

Lifestyle:
- Ocean Drive entertainment
- Lincoln Road shopping
- Beach lifestyle
- 24/7 activity
- High rental potential

Considerations:
- Tourist traffic
- Parking challenges
- Higher insurance costs
- Flood zone requirements""",

            "market_analysis.txt": """SOUTH FLORIDA REAL ESTATE MARKET ANALYSIS 2024

MARKET OVERVIEW
The South Florida real estate market remains resilient with steady demand driven by:
- Continued migration from high-tax states
- International investment
- Limited inventory
- Strong rental market

KEY METRICS (Year-over-Year)

Miami-Dade County:
- Median Sale Price: $450,000 (+6.2%)
- Average Days on Market: 45 (-12%)
- Inventory: 2.8 months supply
- Sales Volume: +3.5%
- Cash Sales: 35% of transactions

Broward County:
- Median Sale Price: $385,000 (+5.8%)
- Average Days on Market: 38
- Inventory: 3.1 months supply

PRICE TRENDS BY PROPERTY TYPE

Single-Family Homes:
- Average appreciation: 7.2% YoY
- Highest demand: 3-4 bedrooms
- Price range most active: $400K-$700K

Condominiums:
- Average appreciation: 4.5% YoY
- Luxury segment (>$1M): +8.3%
- Most active: 2BR units

MARKET DRIVERS

1. Population Growth
- 1,000+ new residents daily
- Tech industry expansion
- Remote work flexibility

2. International Buyers
- 25% of luxury transactions
- Primary markets: Latin America, Europe
- Focus on new construction

3. Rental Market Strength
- Average rent increase: 8% YoY
- Occupancy rate: 96%
- Investor activity high

CHALLENGES

1. Insurance Costs
- Average increase: 40% in 2 years
- Affecting affordability
- Condo associations struggling

2. Interest Rates
- Current average: 7.2%
- Reduced buying power
- Cash buyers advantage

3. Inventory Constraints
- New construction delays
- Existing owners reluctant to sell
- Multiple offer situations common

FORECAST 2024-2025

Expected Trends:
- Price appreciation: 3-5%
- Inventory to remain tight
- Luxury segment outperformance
- Increased condo regulations

Investment Opportunities:
- Rental properties near employment
- Fixer-uppers in transitioning areas
- New construction pre-sales

Risk Factors:
- Hurricane insurance availability
- Sea level rise concerns
- Economic recession possibility""",

            "investment_guide.txt": """REAL ESTATE INVESTMENT STRATEGIES IN SOUTH FLORIDA

RENTAL PROPERTY INVESTMENT

Short-Term Rentals (Airbnb/VRBO):
- Best Areas: South Beach, Brickell, Aventura
- Average ROI: 8-12%
- Regulations: Check city ordinances
- Management: 20-25% of revenue

Long-Term Rentals:
- Target: Near hospitals, universities
- Average ROI: 5-7%
- Tenant screening crucial
- Property management: 8-10%

Key Metrics:
- 1% Rule: Monthly rent = 1% of purchase price
- Cap Rate: 5-7% good in Miami
- Cash-on-cash return: Target 8%+

FIX AND FLIP STRATEGY

Best Neighborhoods:
- Little Havana (gentrifying)
- Allapattah (emerging)
- North Miami (value opportunities)

Typical Timeline:
- Purchase to sale: 4-6 months
- Renovation: 2-3 months
- Holding costs: $3-5K/month

Profit Margins:
- Target: 20-30% ROI
- Average flip profit: $65,000
- Success rate: 70% meet targets

NEW CONSTRUCTION INVESTMENT

Pre-Construction Benefits:
- 10-20% appreciation by completion
- Developer incentives
- Choice units
- Extended payment terms

Risks:
- Delivery delays
- Market changes
- Developer default
- Assessment increases

Due Diligence:
- Developer track record
- Building permits status
- HOA budget projections
- Rental restrictions

COMMERCIAL REAL ESTATE

Retail Opportunities:
- Strip centers in growing areas
- Medical office buildings
- Mixed-use developments

Industrial/Warehouse:
- High demand near ports/airports
- Average cap rate: 6-8%
- Triple net leases preferred

TAX STRATEGIES

Benefits:
- Depreciation deductions
- 1031 exchanges
- Opportunity zones
- Property tax exemptions

Florida Advantages:
- No state income tax
- Homestead exemption
- Foreign investor friendly
- Strong asset protection

FINANCING STRATEGIES

Traditional Lending:
- 20-25% down investment property
- Rates 0.75% higher than primary
- Debt service coverage ratio: 1.25

Alternative Financing:
- Hard money loans (fix/flip)
- Private lenders
- Seller financing
- Partnership structures

Portfolio Building:
- Start with single property
- Reinvest cash flow
- Leverage equity for next purchase
- Diversify property types/locations"""
        }
        
        for filename, content in sample_docs.items():
            filepath = os.path.join(directory_path, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Created sample document: {filename}")
    
    def process_and_store(self, documents):
        """Process documents and store in vector database"""
        if not documents:
            print("No documents to process!")
            return None
            
        # Split documents into chunks
        texts = self.text_splitter.split_documents(documents)
        print(f"Created {len(texts)} text chunks")
        
        # Create or update vector store
        if os.path.exists(self.persist_directory):
            print("Removing existing vector store...")
            shutil.rmtree(self.persist_directory)
            
        vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        print(f"Stored {len(texts)} chunks in vector database")
        return vectorstore


class RAGRetriever:
    """Retrieve relevant information from vector store"""
    
    def __init__(self, persist_directory="./chroma_db"):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.vectorstore = None
        self.persist_directory = persist_directory
        self._initialize_vectorstore()
    
    def _initialize_vectorstore(self):
        """Initialize or create vector store"""
        if os.path.exists(self.persist_directory):
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        else:
            print("Vector store not found. Creating new knowledge base...")
            processor = DocumentProcessor(self.persist_directory)
            documents = processor.load_documents("./real_estate_knowledge")
            self.vectorstore = processor.process_and_store(documents)
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve relevant documents for a query"""
        if not self.vectorstore:
            return []
            
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'relevance_score': 1 / (1 + score)  # Convert distance to similarity
                })
            
            return formatted_results
        except Exception as e:
            print(f"Retrieval error: {e}")
            return []
    
    def get_context(self, query: str, max_tokens: int = 2000) -> str:
        """Get formatted context for the query"""
        results = self.retrieve(query)
        
        if not results:
            return ""
        
        context_parts = []
        token_count = 0
        
        for result in results:
            content = result['content']
            # Approximate token count
            content_tokens = len(content) // 4
            
            if token_count + content_tokens > max_tokens:
                break
            
            # Add source information
            source = result['metadata'].get('source', 'Unknown')
            context_parts.append(f"[From {source}]\n{content}")
            token_count += content_tokens
        
        return "\n\n---\n\n".join(context_parts)


class SerperSearchTool:
    """Tool for searching the internet using Serper API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://google.serper.dev/search"
    
    def search(self, query: str, location: Optional[str] = None, num_results: int = 5) -> Dict[str, Any]:
        """Search the internet for information"""
        headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }
        
        payload = {
            'q': query,
            'num': num_results
        }
        
        if location:
            payload['location'] = location
            
        try:
            response = requests.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}


class ConversationalRealEstateSystem:
    def __init__(self, gemini_api_key: str, serper_api_key: str):
        """Initialize the Conversational Real Estate System with RAG"""
        genai.configure(api_key=gemini_api_key)
        
        self.real_estate_model = genai.GenerativeModel("gemini-2.0-flash")
        
        # Initialize RAG retriever
        print("Initializing knowledge base...")
        self.rag_retriever = RAGRetriever()
        
        # Initialize search tool
        self.search_tool = SerperSearchTool(serper_api_key)
        
        # Initialize smart search decider
        self.search_decider = SmartSearchDecider()
        
        self.json_context = {}
        self.conversation_memory = []
        self.session_file = f"session_{int(time.time())}.json"
        
        self._setup_auto_save()
        
        # Simplified system prompt
        self.system_prompt = """You are Agent Mira, an expert real estate AI assistant with deep knowledge of South Florida real estate.
        
Use the provided knowledge base context to give accurate, specific answers. Be confident and natural in your responses.
Don't mention checking documents or searching - present information as your own expertise.
Focus on being helpful, professional, and conversational."""
    
    def _setup_auto_save(self):
        """Set up automatic saving on exit"""
        atexit.register(self._auto_save)
        signal.signal(signal.SIGINT, self._handle_exit)
        signal.signal(signal.SIGTERM, self._handle_exit)
    
    def _handle_exit(self, signum, frame):
        """Handle exit signals"""
        self._auto_save()
        print(f"\n\nAgent Mira: Thanks for chatting! Feel free to reach out anytime for real estate help. Have a great day!")
        sys.exit(0)
    
    def _auto_save(self):
        """Automatically save session data"""
        try:
            save_data = {
                "conversation_memory": self.conversation_memory,
                "json_context": self.json_context,
                "timestamp": time.time()
            }
            
            with open(self.session_file, 'w') as f:
                json.dump(save_data, f, indent=2)
        except Exception:
            pass

    def _perform_search(self, query: str) -> Dict[str, Any]:
        """Perform search and return results"""
        location = self.json_context.get('location')
        results = self.search_tool.search(query, location=location)
        return results

    def _format_search_results(self, search_results: Dict[str, Any]) -> str:
        """Format search results for context"""
        if 'error' in search_results:
            return ""
        
        formatted_results = []
        
        if 'organic' in search_results:
            for result in search_results['organic'][:3]:
                formatted_results.append(f"{result.get('title', '')}: {result.get('snippet', '')}")
        
        return "\n".join(formatted_results) if formatted_results else ""

    def _get_recent_context(self) -> str:
        """Get recent conversation context"""
        if len(self.conversation_memory) <= 3:
            return json.dumps(self.conversation_memory[-3:])
        else:
            return json.dumps(self.conversation_memory[-3:])

    def _add_to_memory(self, user_msg: str, agent_response: str, sources_used: str = ""):
        """Add exchange to conversation memory"""
        self.conversation_memory.append({
            "user": user_msg,
            "agent": agent_response,
            "timestamp": time.time(),
            "sources": sources_used
        })
        
        # Keep last 20 exchanges
        if len(self.conversation_memory) > 20:
            self.conversation_memory = self.conversation_memory[-20:]

    def update_property_context(self, context_data: Dict[str, Any]):
        """Update property/context information"""
        self.json_context.update(context_data)

    def chat(self, user_input: str) -> str:
        """Enhanced chat method with smart decision making"""
        
        user_input = user_input.strip()
        if not user_input:
            return "I'm here to help! What would you like to know about real estate?"
        
        # Check for exit commands
        if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
            self._auto_save()
            return "Thanks for chatting! Feel free to reach out anytime for real estate help. Have a great day!"
        
        try:
            # Smart decision on information sources
            decision = self.search_decider.analyze_query(user_input, self.rag_retriever)
            
            # Debug info (can be removed in production)
            print(f"\n[Decision: {decision['reasoning']}]")
            
            # Get information from selected sources
            rag_context = ""
            search_context = ""
            sources_used = []
            
            if decision['use_rag']:
                rag_context = self.rag_retriever.get_context(decision['rag_query'])
                if rag_context:
                    sources_used.append("Knowledge Base")
            
            if decision['use_search']:
                # Add location context to search query if not already present
                search_query = decision['search_query']
                if self.json_context.get('location') and self.json_context['location'] not in search_query:
                    search_query += f" {self.json_context['location']}"
                
                search_results = self._perform_search(search_query)
                search_context = self._format_search_results(search_results)
                if search_context:
                    sources_used.append("Current Web Data")
            
            # Combine contexts intelligently
            combined_context = self.search_decider.combine_contexts(
                rag_context, 
                search_context, 
                user_input.lower()
            )
            
            # Build the conversation context
            recent_conversation = ""
            if self.conversation_memory:
                recent_exchanges = self.conversation_memory[-2:]
                for exchange in recent_exchanges:
                    recent_conversation += f"\nUser: {exchange['user']}\nAssistant: {exchange['agent'][:200]}..."
            
            # Construct prompt with all context
            prompt = f"""{self.system_prompt}

Information Sources Used: {', '.join(sources_used) if sources_used else 'General Knowledge'}

{combined_context}

Recent Conversation:
{recent_conversation if recent_conversation else "This is the start of our conversation."}

Current Context: {json.dumps(self.json_context) if self.json_context else "General inquiry"}

User Question: {user_input}

Instructions:
- Provide a comprehensive, natural response using all available information
- Be specific with numbers, prices, and data when available
- If information comes from different time periods, blend them naturally
- Maintain a helpful, professional tone
- Don't mention sources explicitly unless asked"""

            response = self.real_estate_model.generate_content(prompt)
            agent_response = response.text
            
            # Save to memory with sources
            self._add_to_memory(user_input, agent_response, ', '.join(sources_used))
            
            return agent_response
            
        except Exception as e:
            print(f"Error in chat: {e}")
            return "I apologize for the technical issue. Let me try to help you with that question again."


def initialize_knowledge_base():
    """Initialize or update the knowledge base"""
    print("Setting up knowledge base...")
    processor = DocumentProcessor()
    
    # Create knowledge directory if it doesn't exist
    knowledge_dir = "./real_estate_knowledge"
    if not os.path.exists(knowledge_dir):
        os.makedirs(knowledge_dir)
        print(f"Created knowledge directory: {knowledge_dir}")
    
    # Process all documents
    documents = processor.load_documents(knowledge_dir)
    if documents:
        processor.process_and_store(documents)
        print("Knowledge base ready!")
    else:
        print("No documents found in knowledge base.")


def main():
    """Main interactive chat loop"""
    
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    SERPER_API_KEY = os.getenv("SERPER_API_KEY")
    
    if not GEMINI_API_KEY or not SERPER_API_KEY:
        print("Error: Please set both GEMINI_API_KEY and SERPER_API_KEY in your .env file")
        return
    
    try:
        print("Initializing Agentic RAG Bot (Agent Mira)...")
        
        # Initialize knowledge base if needed
        if not os.path.exists("./chroma_db"):
            initialize_knowledge_base()
        
        agent = ConversationalRealEstateSystem(GEMINI_API_KEY, SERPER_API_KEY)
        
        # Set default context
        sample_context = {
            "location": "Miami-Dade County, FL",
            "service_area": "South Florida",
            "specialties": ["residential sales", "investment properties", "luxury homes"],
            "market_focus": "Miami real estate"
        }
        agent.update_property_context(sample_context)
        
        print("\n" + "="*60)
        print("AGENTIC RAG BOT - Your Real Estate AI Expert")
        print("="*60)
        print("\nHi! I'm Agent Mira, your AI-powered real estate expert.")
        print("I combine comprehensive knowledge with real-time data to help you")
        print("with South Florida real estate - neighborhoods, market trends,")
        print("investment strategies, and the home buying process.")
        print("\nFeel free to ask me anything about real estate!")
        print("(Type 'exit' when you're done chatting)")
        print("-" * 60)
        
        # Initial greeting
        greeting = """Hello! I'm Agent Mira, your Agentic RAG-powered real estate assistant. 

I have extensive knowledge about South Florida real estate and can help you with:
• Detailed neighborhood information (Coral Gables, Brickell, Coconut Grove, etc.)
• Current market conditions and trends
• Investment strategies and ROI calculations
• The complete home buying or selling process
• Mortgage and financing options

What aspects of Miami real estate would you like to explore today?"""
        
        print(f"\nAgent Mira: {greeting}\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                response = agent.chat(user_input)
                print(f"\nAgent Mira: {response}\n")
                
                # Check if user said goodbye
                if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                    break
                
            except KeyboardInterrupt:
                agent._handle_exit(None, None)
            except Exception as e:
                print(f"\nAgent Mira: I apologize for the hiccup. What were you asking about?\n")
                
    except Exception as e:
        print(f"Error starting Agent Mira: {e}")
        print("Please check your API keys and try again.")


# Test function for the smart search decider
def test_decision_logic():
    """Test the smart search decision logic with various queries"""
    print("\n" + "="*60)
    print("Testing Smart Search Decision Logic")
    print("="*60)
    
    decider = SmartSearchDecider()
    
    test_queries = [
        "What's the home buying process in Miami?",
        "Current mortgage rates in Miami",
        "Tell me about Coral Gables",
        "What are the latest prices in Brickell?",
        "Compare Brickell vs Aventura for investment",
        "How do I calculate ROI for rental properties?",
        "Properties for sale in Coconut Grove",
        "What amenities does Wynwood offer?",
        "Miami real estate market trends 2024",
        "Tell me about Coral Gables and current market conditions there"
    ]
    
    for query in test_queries:
        decision = decider.analyze_query(query)
        print(f"\nQuery: '{query}'")
        print(f"Decision: RAG={decision['use_rag']}, Search={decision['use_search']}")
        print(f"Reasoning: {decision['reasoning']}")
        print("-" * 40)


if __name__ == "__main__":
    # Uncomment to test decision logic
    # test_decision_logic()
    
    # Run the main chat interface

    main()













