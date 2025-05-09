graph [
  node_attrs_setting [
    name "cpu"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  node_attrs_setting [
    name "gpu"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  node_attrs_setting [
    name "rom"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  link_attrs_setting "_networkx_list_start"
  link_attrs_setting [
    name "bw"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "link"
    type "resource"
  ]
  id 1236
  arrival_time 25540.975305513042
  lifetime 177.19911354871434
  num_nodes 3
  type "path"
  node [
    id 0
    label "0"
    cpu 50
    gpu 8
    rom 25
  ]
  node [
    id 1
    label "1"
    cpu 44
    gpu 26
    rom 28
  ]
  node [
    id 2
    label "2"
    cpu 0
    gpu 17
    rom 21
  ]
  edge [
    source 0
    target 1
    bw 19
  ]
  edge [
    source 1
    target 2
    bw 18
  ]
]
