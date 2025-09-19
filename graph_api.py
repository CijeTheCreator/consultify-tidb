from flask import Flask, request, jsonify
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import sys
import os

# Add the consultify-graphs directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'consultify-graphs'))

from clerking import graph as clerking_graph
from prescriptionAgent import graph as prescription_graph
from models import AgentState, PrescriptionAgentState

app = Flask(__name__)

# Thread pool for async execution
executor = ThreadPoolExecutor(max_workers=4)

def run_graph_async(graph, state):
    """Run a graph asynchronously in a separate thread"""
    try:
        result = graph.invoke(state)
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.route('/clerking/invoke', methods=['POST'])
def invoke_clerking_graph():
    """Endpoint to invoke the clerking graph asynchronously"""
    try:
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Create AgentState from request data
        agent_state = AgentState(**data)
        
        # Submit task to thread pool for async execution
        future = executor.submit(run_graph_async, clerking_graph, agent_state)
        
        # Get task ID (using future object id as simple task ID)
        task_id = str(id(future))
        
        # Store future for later retrieval (in production, use Redis or database)
        app.config.setdefault('tasks', {})[task_id] = future
        
        return jsonify({
            "message": "Clerking graph invocation started",
            "task_id": task_id,
            "status": "running"
        }), 202
        
    except Exception as e:
        return jsonify({"error": f"Failed to invoke clerking graph: {str(e)}"}), 500

@app.route('/prescription/invoke', methods=['POST'])
def invoke_prescription_graph():
    """Endpoint to invoke the prescription agent graph asynchronously"""
    try:
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Create PrescriptionAgentState from request data
        prescription_state = PrescriptionAgentState(**data)
        
        # Submit task to thread pool for async execution
        future = executor.submit(run_graph_async, prescription_graph, prescription_state)
        
        # Get task ID (using future object id as simple task ID)
        task_id = str(id(future))
        
        # Store future for later retrieval (in production, use Redis or database)
        app.config.setdefault('tasks', {})[task_id] = future
        
        return jsonify({
            "message": "Prescription graph invocation started",
            "task_id": task_id,
            "status": "running"
        }), 202
        
    except Exception as e:
        return jsonify({"error": f"Failed to invoke prescription graph: {str(e)}"}), 500

@app.route('/task/<task_id>/status', methods=['GET'])
def get_task_status(task_id):
    """Get the status of an async task"""
    try:
        tasks = app.config.get('tasks', {})
        future = tasks.get(task_id)
        
        if not future:
            return jsonify({"error": "Task not found"}), 404
        
        if future.done():
            result = future.result()
            # Clean up completed task
            del tasks[task_id]
            
            if result["success"]:
                return jsonify({
                    "task_id": task_id,
                    "status": "completed",
                    "result": result["result"]
                })
            else:
                return jsonify({
                    "task_id": task_id,
                    "status": "failed",
                    "error": result["error"]
                }), 500
        else:
            return jsonify({
                "task_id": task_id,
                "status": "running"
            })
            
    except Exception as e:
        return jsonify({"error": f"Failed to get task status: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "Graph API is running"})

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)