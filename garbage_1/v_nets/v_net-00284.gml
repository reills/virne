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
  id 284
  arrival_time 5522.771641899012
  lifetime 259.0881799814133
  num_nodes 7
  type "path"
  node [
    id 0
    label "0"
    cpu 7
    gpu 36
    rom 33
  ]
  node [
    id 1
    label "1"
    cpu 6
    gpu 37
    rom 35
  ]
  node [
    id 2
    label "2"
    cpu 33
    gpu 8
    rom 35
  ]
  node [
    id 3
    label "3"
    cpu 7
    gpu 21
    rom 27
  ]
  node [
    id 4
    label "4"
    cpu 42
    gpu 48
    rom 27
  ]
  node [
    id 5
    label "5"
    cpu 29
    gpu 31
    rom 22
  ]
  node [
    id 6
    label "6"
    cpu 26
    gpu 4
    rom 20
  ]
  edge [
    source 0
    target 1
    bw 19
  ]
  edge [
    source 1
    target 2
    bw 48
  ]
  edge [
    source 2
    target 3
    bw 42
  ]
  edge [
    source 3
    target 4
    bw 31
  ]
  edge [
    source 4
    target 5
    bw 30
  ]
  edge [
    source 5
    target 6
    bw 19
  ]
]
